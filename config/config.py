import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MODELS_DIR = os.path.join(DATA_DIR, "models")
EXPERIMENTS_DIR = os.path.join(DATA_DIR, "experiments")
OUTPUT_DIR = "outputs"
DATASETS_DIR = "datasets"


for directory in [
    DATA_DIR,
    IMAGES_DIR,
    MODELS_DIR,
    EXPERIMENTS_DIR,
    OUTPUT_DIR,
    DATASETS_DIR,
]:
    os.makedirs(directory, exist_ok=True)


PLOT_STYLE = "ggplot"
FIGURE_SIZE = [9, 4]
FONT_SIZE = 12
LABEL_SIZE = 14
TITLE_SIZE = 16


DATASET_PATH = "datasets/dataset_ip_classified.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42


SCORE_COLS = [
    "abuseipdb_confidence_score",
    "abuseipdb_total_reports",
    "abuseipdb_num_distinct_users",
    "ipvoid_detection_count",
    "risk_recommended_pulsedive",
    "virustotal_reputation",
    "virustotal_harmless",
    "virustotal_malicious",
    "virustotal_undetected",
    "virustotal_suspicious",
]


PARAM_GRID = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [8, 10, 12],
        "min_samples_split": [2, 4],
        "max_features": ["sqrt", "log2"],
        "min_samples_leaf": [3, 5, 10],
    },
    "SVM": {
        "C": [0.01, 0.1, 1],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
    },
    "Neural Network": {
        "hidden_layer_sizes": [(50,), (50, 25)],
        "alpha": [0.001, 0.01],
        "learning_rate": ["adaptive"],
    },
    "Extra Trees": {
        "n_estimators": [100, 200],
        "max_depth": [10, 12, 15],
        "min_samples_split": [2, 4],
    },
    "Decision Tree": {
        "max_depth": [6, 8, 10],
        "min_samples_split": [2, 4],
        "min_samples_leaf": [3, 5, 10],
        "max_features": ["sqrt", "log2"],
    },
    "KNN": {
        "pca__n_components": [0.65, 0.70, 0.75],
        "knn__n_neighbors": [31, 33, 35],
        "knn__weights": ["uniform"],
        "knn__metric": ["euclidean"],
        "knn__p": [2],
        "knn__leaf_size": [50],
    },
}


IP_CLASSIFICATION_CONFIG = {
    "input_file": "datasets/dataset_ip.csv",
    "output_file": "datasets/dataset_cal_w.csv",
    "dataset_path": "datasets/dataset_ip.csv",
    "output_path": "datasets/novo_dataset_com_pesos.csv",
    "feature_ranges": {
        "abuseipdb_confidence_score": [0, 100],
        "abuseipdb_total_reports": [0, 85000],
        "abuseipdb_num_distinct_users": [0, 2000],
        "ipvoid_detection_count": [0, 40],
        "risk_recommended_pulsedive": [0, 5],
        "virustotal_reputation": [-127, 565],
        "virustotal_harmless": [0, 86],
        "virustotal_malicious": [0, 20],
        "virustotal_undetected": [0, 91],
        "virustotal_suspicious": [0, 5],
    },
    "inverse_features": ["virustotal_reputation", "virustotal_harmless"],
}


CLASSIFICATION_THRESHOLDS = {
    "allowlist_max": 0.2,
    "suspicious_max": 0.4,
    "labels": ["Allowlist", "Suspicious", "Denylist"],
}


DEFAULT_EXPERIMENT_CONFIG = {
    "data_file": DATASET_PATH,
    "test_size": TEST_SIZE,
    "random_state": RANDOM_STATE,
    "score_cols": SCORE_COLS,
    "use_smote": False,
    "corrected_pipeline": True,
}
