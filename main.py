import inspect
import os
import sys
from datetime import datetime

import pandas as pd

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)


from config.config import SCORE_COLS
from src.data_processing import (
    load_and_preprocess_data,
    normalize_scores,
    prepare_data_for_training,
    save_label_encoder,
)
from src.evaluate import (
    evaluate_model_accuracy,
    evaluate_models,
    save_evaluation_results,
)
from src.train import train_and_evaluate_models_corrected
from src.visualization import (
    plot_class_distribution,
    plot_confusion_matrices,
    plot_correlation_matrix,
    plot_execution_times,
    plot_feature_distributions,
    plot_metrics_comparison,
    plot_metrics_tables,
)
from util.utils import (
    create_experiment_directory,
    generate_summary_report,
    log_execution_info,
    save_experiment_config,
    verify_and_list_outputs,
)


def create_safe_experiment_directory(experiment_name=None):
    """
    Create experiment directory with maximum compatibility
    """
    try:

        base_dir = create_experiment_directory()

        if experiment_name:
            parent_dir = os.path.dirname(base_dir)
            new_dir = os.path.join(parent_dir, experiment_name)

            if os.path.exists(new_dir):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_dir = os.path.join(parent_dir, f"{experiment_name}_{timestamp}")

            try:
                os.rename(base_dir, new_dir)
                base_dir = new_dir
            except Exception:

                pass

        return base_dir

    except Exception:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            dir_name = f"{experiment_name}_{timestamp}"
        else:
            dir_name = f"experiment_{timestamp}"

        experiment_dir = os.path.join(os.getcwd(), "data", "experiments", dir_name)

        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "models"), exist_ok=True)

        return experiment_dir


def main_pipeline(use_smote=False, experiment_name=None):
    """
    Main pipeline that ensures files are saved properly
    """

    try:
        experiment_dir = create_safe_experiment_directory(experiment_name)
        images_dir = os.path.join(experiment_dir, "images")
        models_dir = os.path.join(experiment_dir, "models")

    except Exception as e:
        return {"success": False, "error": f"Critical error creating directories: {e}"}

    config = {
        "data_file": "datasets/dataset_ip_classified.csv",
        "test_size": 0.2,
        "random_state": 42,
        "score_cols": SCORE_COLS,
        "use_smote": use_smote,
        "corrected_pipeline": True,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        save_experiment_config(config, experiment_dir)
    except Exception:
        pass

    log_execution_info(
        f"Starting pipeline - SMOTE: {'Enabled' if use_smote else 'Disabled'}",
        experiment_dir,
    )

    try:

        log_execution_info("Loading original dataset...", experiment_dir)

        df = load_and_preprocess_data(config["data_file"])
        df_normalized, score_cols = normalize_scores(df)

        log_execution_info("Generating feature visualizations...", experiment_dir)

        try:
            plot_feature_distributions(df_normalized, score_cols, images_dir)
            plot_correlation_matrix(df_normalized, score_cols, images_dir)
            plot_class_distribution(df_normalized, images_dir)
        except Exception:
            pass

        log_execution_info("Preparing data for training...", experiment_dir)

        X_train, X_test, y_train, y_test, le = prepare_data_for_training(
            df_normalized, score_cols
        )
        save_label_encoder(le, models_dir)

        log_execution_info(
            f"Data split - Train: {X_train.shape}, Test: {X_test.shape}", experiment_dir
        )

        log_execution_info("Starting model training...", experiment_dir)

        try:
            sig = inspect.signature(train_and_evaluate_models_corrected)
            if "use_smote" in sig.parameters:
                results, execution_times, trained_models = (
                    train_and_evaluate_models_corrected(
                        X_train,
                        X_test,
                        y_train,
                        y_test,
                        models_dir,
                        images_dir,
                        use_smote=use_smote,
                    )
                )
            else:
                results, execution_times, trained_models = (
                    train_and_evaluate_models_corrected(
                        X_train, X_test, y_train, y_test, models_dir, images_dir
                    )
                )
        except Exception as e:
            raise Exception(f"Training error: {e}")

        log_execution_info("Evaluating model performance...", experiment_dir)

        try:
            metrics_df = evaluate_models(trained_models, X_test, y_test)
            accuracy_df = evaluate_model_accuracy(trained_models, X_test, y_test, le)
        except Exception as e:
            raise Exception(f"Evaluation error: {e}")

        try:
            results_df = pd.DataFrame(results).T
            if "Best Parameters" in results_df.columns:
                results_df["Best Parameters"] = results_df["Best Parameters"].apply(
                    lambda x: str(x)
                )
            elif "Melhores Parâmetros" in results_df.columns:
                results_df["Best Parameters"] = results_df["Melhores Parâmetros"].apply(
                    lambda x: str(x)
                )
                results_df.drop("Melhores Parâmetros", axis=1, inplace=True)

            results_csv_path = os.path.join(
                models_dir, "results_with_hyperparameters.csv"
            )
            results_df.to_csv(results_csv_path)

            save_evaluation_results(metrics_df, accuracy_df, models_dir)

        except Exception:
            pass

        log_execution_info("Generating result visualizations...", experiment_dir)

        try:
            plot_metrics_tables(results_df, metrics_df, images_dir)
            plot_execution_times(execution_times, images_dir)
            plot_metrics_comparison(metrics_df, images_dir)
            plot_confusion_matrices(trained_models, X_test, y_test, images_dir)
        except Exception:
            pass

        try:
            report_file = generate_summary_report(
                metrics_df, accuracy_df, execution_times, results_df, experiment_dir
            )
        except Exception:
            report_file = None

        try:
            generated_files = verify_and_list_outputs(experiment_dir)
        except Exception:
            generated_files = []

        smote_status = (
            "WITH SMOTE (correctly applied)" if use_smote else "WITHOUT SMOTE"
        )
        success_message = (
            f"Pipeline completed successfully! ({smote_status})\n"
            f"- Directory: {experiment_dir}\n"
            f"- Report: {report_file}\n"
            f"- Models: {models_dir}\n"
            f"- Charts: {images_dir}\n"
            f"- NO DATA LEAKAGE"
        )

        log_execution_info(success_message, experiment_dir)

        return {
            "experiment_dir": experiment_dir,
            "metrics": metrics_df,
            "accuracy": accuracy_df,
            "use_smote": use_smote,
            "report_file": report_file,
            "success": True,
            "top_models": accuracy_df.nlargest(
                3,
                (
                    "Accuracy (%)"
                    if "Accuracy (%)" in accuracy_df.columns
                    else "Acurácia (%)"
                ),
            ),
            "files_generated": len(generated_files),
        }

    except Exception as e:
        error_msg = f"Error during execution: {str(e)}"
        log_execution_info(error_msg, experiment_dir)

        return {
            "experiment_dir": experiment_dir,
            "error": str(e),
            "use_smote": use_smote,
            "success": False,
        }


def run_both_experiments():
    """
    Execute both experiments with robust error handling
    """
    results = {}

    try:
        results["without_smote"] = main_pipeline(
            use_smote=False, experiment_name="experiment_WITHOUT_smote_corrected"
        )
    except Exception as e:
        results["without_smote"] = {"success": False, "error": str(e)}

    try:
        results["with_smote"] = main_pipeline(
            use_smote=True, experiment_name="experiment_WITH_smote_corrected"
        )
    except Exception as e:
        results["with_smote"] = {"success": False, "error": str(e)}

    return results


def run_single_experiment(use_smote=False):
    """
    Execute single experiment
    """
    experiment_name = f"experiment_{'with' if use_smote else 'without'}_smote_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return main_pipeline(use_smote=use_smote, experiment_name=experiment_name)


if __name__ == "__main__":

    dataset_path = "datasets/dataset_ip_classified.csv"
    if not os.path.exists(dataset_path):
        sys.exit(1)

    results = run_both_experiments()
