import json
import os
import shutil
from datetime import datetime

import pandas as pd


def create_experiment_directory():
    """
    Create experiment directory structure with robust verification.
    Ensures directories are created and visible.
    """
    if __file__:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    else:
        base_dir = os.getcwd()

    if not os.access(base_dir, os.W_OK):
        base_dir = os.getcwd()

    data_dir = os.path.join(base_dir, "data")
    images_dir = os.path.join(data_dir, "images")
    models_dir = os.path.join(data_dir, "models")
    experiments_dir = os.path.join(data_dir, "experiments")

    dirs_to_create = [data_dir, images_dir, models_dir, experiments_dir]

    for directory in dirs_to_create:
        try:
            os.makedirs(directory, exist_ok=True)

            if not os.path.exists(directory):
                raise Exception(f"Directory not created: {directory}")

        except Exception as e:

            fallback_dir = os.path.join(os.getcwd(), os.path.basename(directory))
            os.makedirs(fallback_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(experiments_dir, f"experiment_{timestamp}")
    experiment_images = os.path.join(experiment_dir, "images")
    experiment_models = os.path.join(experiment_dir, "models")

    experiment_dirs = [experiment_dir, experiment_images, experiment_models]

    for directory in experiment_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(directory):
                raise Exception(f"Experiment directory not created: {directory}")
        except Exception as e:

            fallback_path = os.path.join(os.getcwd(), f"experiment_{timestamp}")
            os.makedirs(fallback_path, exist_ok=True)
            fallback_images = os.path.join(fallback_path, "images")
            fallback_models = os.path.join(fallback_path, "models")
            os.makedirs(fallback_images, exist_ok=True)
            os.makedirs(fallback_models, exist_ok=True)

            return fallback_path

    if not os.path.exists(experiment_dir):
        raise Exception(
            f"CRITICAL FAILURE: Experiment directory does not exist: {experiment_dir}"
        )

    try:
        test_file = os.path.join(experiment_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("write test")
        os.remove(test_file)

    except Exception as e:
        raise

    return experiment_dir


def save_experiment_config(config_dict, experiment_dir):
    """
    Save experiment configuration with verification.
    """
    try:
        config_path = os.path.join(experiment_dir, "experiment_config.json")

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        if not os.path.exists(config_path):
            raise Exception("Configuration file was not created")

    except Exception as e:

        fallback_path = os.path.join(os.getcwd(), "experiment_config.json")
        try:
            with open(fallback_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=4, ensure_ascii=False)
        except:
            pass


def log_execution_info(message, experiment_dir, print_msg=False):
    """
    Execution log with robust fallback.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"

    try:
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)

        log_path = os.path.join(experiment_dir, "execution.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_message)
    except Exception as e:

        try:
            log_path = os.path.join(os.getcwd(), "execution.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[FALLBACK] {log_message}")
        except:
            pass


def generate_summary_report(
    metrics_df, accuracy_df, execution_times, results_df, experiment_dir
):
    """
    Generate summary report with save verification.
    """
    try:
        report = []
        report.append("# Model Execution Report\n")
        report.append(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("\n## Optimized Hyperparameters by Model\n")
        for model_name, row in results_df.iterrows():

            hyperparams = row.get(
                "Best Parameters", row.get("Melhores Parâmetros", "N/A")
            )
            report.append(f"- **{model_name}**: {hyperparams}")

        report.append("\n## Performance Metrics\n")
        report.append(metrics_df.to_markdown())

        report.append("\n## Accuracy by Model\n")
        report.append(accuracy_df.to_markdown())

        report.append("\n## Execution Times\n")
        exec_df = pd.DataFrame.from_dict(
            execution_times, orient="index", columns=["Time (s)"]
        )
        report.append(exec_df.to_markdown())

        report_content = "\n".join(report)

        try:
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir, exist_ok=True)

            report_path = os.path.join(experiment_dir, "summary_report.md")
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            if os.path.exists(report_path) and os.path.getsize(report_path) > 0:
                return report_path
            else:
                raise Exception("Report was not saved correctly")

        except Exception as e:

            fallback_path = os.path.join(
                os.getcwd(),
                f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            )
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            return fallback_path

    except Exception as e:
        return None


def copy_to_main_directories(file_path, data_dir, file_type):
    """
    Copy files to main directories with verification.
    """
    if not os.path.exists(file_path):
        return

    try:
        filename = os.path.basename(file_path)
        if file_type == "image":
            dest_dir = os.path.join(data_dir, "images")
        else:
            dest_dir = os.path.join(data_dir, "models")

        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(file_path, dest_path)

    except Exception as e:
        pass


def verify_and_list_outputs(experiment_dir):
    """
    Verify and list all files generated in the experiment.
    """
    if not os.path.exists(experiment_dir):
        return []

    all_files = []
    for root, dirs, files in os.walk(experiment_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            rel_path = os.path.relpath(file_path, experiment_dir)
            all_files.append((rel_path, file_size))

    return all_files
