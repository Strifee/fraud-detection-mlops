import mlflow
import os


def setup_mlflow(experiment_name: str = "fraud-detection-comparison"):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"Experiment           : {experiment_name}")
    print(f"MLflow UI            : run 'mlflow ui' then open http://localhost:5000")
    return mlflow.get_experiment_by_name(experiment_name)


def get_best_run(experiment_name: str = "fraud-detection-comparison", metric: str = "test_auc_roc"):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"No experiment found: {experiment_name}")
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1
    )

    if not runs:
        print("No runs found.")
        return None

    best = runs[0]
    print(f"\nBest run: {best.info.run_name}")
    print(f"  Run ID  : {best.info.run_id}")
    print(f"  AUC-ROC : {best.data.metrics.get('test_auc_roc', 'N/A')}")
    print(f"  F1      : {best.data.metrics.get('test_f1', 'N/A')}")
    print(f"  Recall  : {best.data.metrics.get('test_recall', 'N/A')}")
    return best


if __name__ == "__main__":
    setup_mlflow()
    get_best_run()
