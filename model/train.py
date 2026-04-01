import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    f1_score
)
import xgboost as xgb
import mlflow
import mlflow.pytorch
import mlflow.xgboost
import argparse
import logging
import time
from model import FraudDetectorNN, FraudDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NN_CONFIG = {
    "epochs": 30,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "dropout": 0.3,
    "weight_decay": 1e-4,
}

XGBOOST_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 10,
    "eval_metric": "auc",
    "use_label_encoder": False,
}


def load_splits():
    X_train = np.load("data/processed/X_train.npy")
    X_val   = np.load("data/processed/X_val.npy")
    X_test  = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_val   = np.load("data/processed/y_val.npy")
    y_test  = np.load("data/processed/y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test




def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        "auc_roc":   round(roc_auc_score(y_true, y_pred_proba), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
    }



def train_pytorch(X_train, X_val, X_test, y_train, y_val, y_test):
    logger.info("Training Neural Network...")

    with mlflow.start_run(run_name="NN_FraudDetector"):
        mlflow.log_params(NN_CONFIG)

        train_ds = FraudDataset(X_train, y_train)
        val_ds   = FraudDataset(X_val, y_val)
        train_dl = DataLoader(train_ds, batch_size=NN_CONFIG["batch_size"], shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=NN_CONFIG["batch_size"])

        model = FraudDetectorNN(input_dim=X_train.shape[1], dropout=NN_CONFIG["dropout"]).to(DEVICE)

        pos_weight = torch.tensor([10.0]).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer  = torch.optim.Adam(
            model.parameters(),
            lr=NN_CONFIG["learning_rate"],
            weight_decay=NN_CONFIG["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        best_val_auc = 0
        start_time   = time.time()

        for epoch in range(NN_CONFIG["epochs"]):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_dl:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                preds = model(X_batch).squeeze()
                loss  = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_batch, _ in val_dl:
                    out = model(X_batch.to(DEVICE)).squeeze()
                    val_preds.extend(out.cpu().numpy())

            val_metrics = compute_metrics(y_val, np.array(val_preds))
            scheduler.step(1 - val_metrics["auc_roc"])

            mlflow.log_metrics({
                "train_loss": train_loss / len(train_dl),
                **{f"val_{k}": v for k, v in val_metrics.items()}
            }, step=epoch)

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                torch.save(model.state_dict(), "model/best_pytorch_model.pt")

            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{NN_CONFIG['epochs']} — "
                           f"Loss: {train_loss/len(train_dl):.4f} — "
                           f"Val AUC: {val_metrics['auc_roc']:.4f}")


        model.load_state_dict(torch.load("model/best_pytorch_model.pt"))
        model.eval()
        test_ds = FraudDataset(X_test, y_test)
        test_dl = DataLoader(test_ds, batch_size=NN_CONFIG["batch_size"])
        test_preds = []
        with torch.no_grad():
            for X_batch, _ in test_dl:
                out = model(X_batch.to(DEVICE)).squeeze()
                test_preds.extend(out.cpu().numpy())

        test_metrics = compute_metrics(y_test, np.array(test_preds))
        training_time = round(time.time() - start_time, 2)

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.pytorch.log_model(model, "pytorch_model")

        logger.info(f"\n{'='*50}")
        logger.info("PyTorch Test Results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k.upper()}: {v}")
        logger.info(f"  Training time: {training_time}s")
        logger.info(f"{'='*50}\n")

        return test_metrics, model




def train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test):
    logger.info("Training XGBoost...")

    with mlflow.start_run(run_name="XGBoost_FraudDetector"):
        mlflow.log_params(XGBOOST_CONFIG)

        start_time = time.time()

        model = xgb.XGBClassifier(**XGBOOST_CONFIG)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50
        )

        test_preds = model.predict_proba(X_test)[:, 1]
        test_metrics  = compute_metrics(y_test, test_preds)
        training_time = round(time.time() - start_time, 2)

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metric("training_time_seconds", training_time)
        mlflow.xgboost.log_model(model, "xgboost_model")

        logger.info(f"\n{'='*50}")
        logger.info("XGBoost Test Results:")
        for k, v in test_metrics.items():
            logger.info(f"  {k.upper()}: {v}")
        logger.info(f"  Training time: {training_time}s")
        logger.info(f"{'='*50}\n")

        return test_metrics, model



def print_comparison(pytorch_metrics, xgboost_metrics):
    print("\n" + "="*60)
    print("MODEL COMPARISON — Test Set Results")
    print("="*60)
    print(f"{'Metric':<15} {'PyTorch NN':>15} {'XGBoost':>15}")
    print("-"*60)
    for metric in ["auc_roc", "precision", "recall", "f1"]:
        pt  = pytorch_metrics[metric]
        xgb = xgboost_metrics[metric]
        winner = "✓" if pt > xgb else " "
        print(f"{metric:<15} {pt:>14} {winner}  {xgb:>14}")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["pytorch", "xgboost", "both"], default="both")
    args = parser.parse_args()

    mlflow.set_experiment("fraud-detection-comparison")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    logger.info(f"Running on: {DEVICE}")

    pytorch_metrics, xgboost_metrics = None, None

    if args.model in ["pytorch", "both"]:
        pytorch_metrics, _ = train_pytorch(X_train, X_val, X_test, y_train, y_val, y_test)

    if args.model in ["xgboost", "both"]:
        xgboost_metrics, _ = train_xgboost(X_train, X_val, X_test, y_train, y_val, y_test)

    if args.model == "both" and pytorch_metrics and xgboost_metrics:
        print_comparison(pytorch_metrics, xgboost_metrics)


## TODO: Ensemble Model
## Add a combined PyTorch NN + XGBoost + Isolation Forest 
## using majority voting to improve recall