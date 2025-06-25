import os
import time

import joblib
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config.config import PARAM_GRID

from .models import get_models
from .visualization import plot_learning_curves, plot_learning_curves_cnn


def train_and_evaluate_models_corrected(
    X_train, X_test, y_train, y_test, models_dir, images_dir, use_smote=False
):
    """
    Train and evaluate models WITH DATA LEAKAGE CORRECTION.

    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training labels
        y_test (pd.Series): Test labels
        models_dir (str): Directory to save models
        images_dir (str): Directory to save images
        use_smote (bool): Whether to apply SMOTE to training set

    Returns:
        tuple: (results, execution_times, trained_models)
    """

    if use_smote:

        smote = SMOTE(random_state=42)
        X_train_processed, y_train_processed = smote.fit_resample(X_train, y_train)

    else:
        X_train_processed = X_train.copy()
        y_train_processed = y_train.copy()

    input_shape = (X_train_processed.shape[1], 1)

    try:
        models = get_models(input_shape, use_smote=use_smote)
    except TypeError:

        models = get_models_adapted(input_shape, use_smote)

    results = {}
    execution_times = {}
    trained_models = {}

    for name, model in models.items():
        start_time = time.time()

        if name == "CNN":
            model, hyperparameters = model

            X_train_reshaped = X_train_processed.values.reshape(
                X_train_processed.shape[0], X_train_processed.shape[1], 1
            )
            X_test_reshaped = X_test.values.reshape(X_test.shape[0], X_test.shape[1], 1)

            history = model.fit(
                X_train_reshaped,
                y_train_processed,
                epochs=hyperparameters["epochs"],
                batch_size=hyperparameters["batch_size"],
                validation_split=0.2,
                verbose=0,
            )

            train_score = model.evaluate(
                X_train_reshaped, y_train_processed, verbose=0
            )[1]
            test_score = model.evaluate(X_test_reshaped, y_test, verbose=0)[1]

            results[name] = {
                "Best Parameters": hyperparameters,
                "Train Score": train_score,
                "Test Score": test_score,
            }
            trained_models[name] = model

            plot_learning_curves_cnn(history, name, images_dir)

            model.save(os.path.join(models_dir, f"CNN_model.keras"))

        else:

            grid_search = GridSearchCV(
                model, PARAM_GRID[name], cv=5, scoring="f1_weighted", n_jobs=-1
            )

            try:
                grid_search.fit(X_train_processed, y_train_processed)
            except Exception as e:
                continue

            train_score = grid_search.score(X_train_processed, y_train_processed)
            test_score = grid_search.score(X_test, y_test)

            results[name] = {
                "Best Parameters": grid_search.best_params_,
                "Train Score": train_score,
                "Test Score": test_score,
            }
            trained_models[name] = grid_search.best_estimator_

            plot_learning_curves(
                grid_search.best_estimator_,
                X_train_processed,
                y_train_processed,
                name,
                images_dir,
            )

            joblib.dump(
                grid_search.best_estimator_,
                os.path.join(models_dir, f"{name}_model.joblib"),
            )

        execution_time = time.time() - start_time
        execution_times[name] = execution_time

    if not os.path.exists(os.path.join(models_dir, "label_encoder.joblib")):
        le = LabelEncoder()
        le.fit(y_train_processed)
        joblib.dump(le, os.path.join(models_dir, "label_encoder.joblib"))

    return results, execution_times, trained_models


def get_models_adapted(input_shape, use_smote):
    """
    Adapted function to use original models.py with correct configurations.
    """

    def create_cnn_model_local(input_shape):
        hyperparameters = {
            "conv1_filters": 32,
            "conv2_filters": 64,
            "kernel_size": 3,
            "dense1_units": 32,
            "dense2_units": 16,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 10,
        }

        model = Sequential(
            [
                Input(shape=input_shape),
                Conv1D(
                    hyperparameters["conv1_filters"],
                    kernel_size=hyperparameters["kernel_size"],
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                Conv1D(
                    hyperparameters["conv2_filters"],
                    kernel_size=hyperparameters["kernel_size"],
                    padding="same",
                    activation="relu",
                ),
                BatchNormalization(),
                Flatten(),
                Dense(hyperparameters["dense1_units"], activation="relu"),
                Dropout(hyperparameters["dropout_rate"]),
                Dense(hyperparameters["dense2_units"], activation="relu"),
                Dense(3, activation="softmax"),
            ]
        )

        model.compile(
            optimizer=Adam(learning_rate=hyperparameters["learning_rate"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model, hyperparameters

    cnn_model, cnn_params = create_cnn_model_local(input_shape)

    class_weight = None if use_smote else "balanced"

    return {
        "Random Forest": RandomForestClassifier(
            class_weight=class_weight, random_state=42
        ),
        "SVM": SVC(class_weight=class_weight, probability=True, random_state=42),
        "Neural Network": MLPClassifier(
            max_iter=1000,
            random_state=42,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            hidden_layer_sizes=(100, 50),
            activation="relu",
            solver="adam",
            batch_size="auto",
            shuffle=True,
            verbose=False,
        ),
        "Extra Trees": ExtraTreesClassifier(class_weight=class_weight, random_state=42),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weight, random_state=42
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("pca", PCA()),
                ("knn", KNeighborsClassifier()),
            ]
        ),
        "CNN": (cnn_model, cnn_params),
    }


def train_and_evaluate_models_with_balancing(
    X_train, X_test, y_train, y_test, models_dir, images_dir
):
    """
    Compatibility function with old code.
    DEPRECATED: Use train_and_evaluate_models_corrected() with use_smote parameter.
    """
    return train_and_evaluate_models_corrected(
        X_train, X_test, y_train, y_test, models_dir, images_dir, use_smote=False
    )
