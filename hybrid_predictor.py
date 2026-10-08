#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hybrid Machine Learning Model for Bacterial smORF Prediction
=============================================================

This repository contains the implementation of a hybrid ensemble model
combining Random Forest (RF) and Gradient Boosting (GB) classifiers to
predict small open reading frames (smORFs) in bacterial genomes, using
Mycobacterium tuberculosis as the model organism.

The pipeline:

    1. Downloads the M. tuberculosis dataset from the linked raw URL.
    2. Cleans sequences to retain only A, C, G, T characters.
    3. Extracts character-level 6-mer features using CountVectorizer
       (min_df=5, fitted on the training split only).
    4. Trains the hybrid ensemble (RF + GB, soft voting) on 80% of the data.
    5. Reports 10-fold cross-validation and hold-out test performance
       with 95% confidence intervals.
    6. Saves the fitted models and vectorizer for reuse.

Requirements
------------
Python 3.7 or later, with pandas, numpy, scikit-learn, scipy,
matplotlib, and joblib installed.

Usage
-----
python hybrid_smorf_predictor.py

Expected runtime is approximately 30 minutes on a modern laptop.
Output files are written to the results directory.
"""

import os
import re
import math
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["figure.dpi"] = 300


# ============================================================
# Configuration
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/focuzguy1/"
    "Hybrid-ML-Prediction/refs/heads/main/Mtb_positive_negative_data.csv"
)

MODEL_PATH = "hybrid_smorf_model.joblib"
RESULTS_DIR = "results"

KMER_SIZE = 6
MIN_DF = 5

RF_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

GB_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=5,
    subsample=0.8,
    validation_fraction=0.1,
    n_iter_no_change=20,
    tol=1e-4,
    random_state=42,
)

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 10
CONFIDENCE = 0.95


# ============================================================
# Data preparation
# ============================================================

def clean_sequence(seq):
    """Keep only A, C, G, T characters and convert to uppercase."""
    seq = str(seq).upper().strip()
    return re.sub(r"[^ACGT]", "", seq)


def load_data():
    """Download the dataset and return cleaned sequences and labels."""
    df = pd.read_csv(DATA_URL, sep="\t")
    df.columns = df.columns.str.strip()
    df["sequence"] = df["sequence"].apply(clean_sequence)
    return df["sequence"].values, df["class"].values


def vectorize(X_train, X_test):
    """Fit CountVectorizer on the training split and transform both splits."""
    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(KMER_SIZE, KMER_SIZE),
        min_df=MIN_DF,
    )
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    return X_train_vect, X_test_vect, vectorizer


# ============================================================
# Model construction
# ============================================================

def build_model():
    """Return an unfitted VotingClassifier combining RF and GB with soft voting."""
    rf = RandomForestClassifier(**RF_PARAMS)
    gb = GradientBoostingClassifier(**GB_PARAMS)
    return VotingClassifier(estimators=[("rf", rf), ("gb", gb)], voting="soft")


def fit_model(model, X_train, y_train):
    """
    Fit the RF and GB classifiers with balanced class weights.

    Random Forest applies class_weight='balanced' internally.
    Gradient Boosting receives balanced sample weights through the fit call.
    """
    sw = compute_sample_weight("balanced", y_train)
    rf = model.named_estimators_["rf"]
    gb = model.named_estimators_["gb"]
    rf.fit(X_train, y_train)
    gb.fit(X_train, y_train, sample_weight=sw)
    return rf, gb


def predict_soft(rf, gb, X):
    """Average the class probabilities of RF and GB, and threshold at 0.5."""
    p_rf = rf.predict_proba(X)[:, 1]
    p_gb = gb.predict_proba(X)[:, 1]
    p_avg = (p_rf + p_gb) / 2
    return (p_avg >= 0.5).astype(int), p_avg


# ============================================================
# Evaluation
# ============================================================

def wilson_ci(k, n, confidence=CONFIDENCE):
    """Wilson score interval for a proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcc_ci(mcc, n, confidence=CONFIDENCE):
    """Normal approximation confidence interval for MCC."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    se = math.sqrt((1 - mcc**2) / (n - 1))
    return max(-1.0, mcc - z * se), min(1.0, mcc + z * se)


def auc_ci(auc_score, n_pos, n_neg, confidence=CONFIDENCE):
    """Hanley-McNeil confidence interval for AUC."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    q1 = auc_score / (2 - auc_score)
    q2 = (2 * auc_score**2) / (1 + auc_score)
    se = math.sqrt(
        (auc_score * (1 - auc_score)
         + (n_pos - 1) * (q1 - auc_score**2)
         + (n_neg - 1) * (q2 - auc_score**2))
        / (n_pos * n_neg)
    )
    return max(0.0, auc_score - z * se), min(1.0, auc_score + z * se), se


def evaluate(y_true, y_pred, y_proba):
    """Compute positive-class metrics with 95% confidence intervals."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average="binary", zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_proba)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    acc_lo, acc_hi = wilson_ci(tp + tn, len(y_true))
    mcc_lo, mcc_hi = mcc_ci(mcc, len(y_true))
    auc_lo, auc_hi, auc_se = auc_ci(auc_score, n_pos, n_neg)

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    return {
        "accuracy": accuracy,
        "accuracy_ci": (acc_lo, acc_hi),
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
        "mcc_ci": (mcc_lo, mcc_hi),
        "auc": auc_score,
        "auc_ci": (auc_lo, auc_hi),
        "auc_se": auc_se,
        "confusion": cm,
        "fpr": fpr,
        "tpr": tpr,
        "n_total": len(y_true),
        "n_pos": n_pos,
        "n_neg": n_neg,
    }


def print_metrics(metrics, label):
    """Print a formatted metrics block."""
    print()
    print("=" * 70)
    print("Performance: {0}".format(label))
    print("=" * 70)
    print("Samples: {0} ({1} positive, {2} negative)".format(
        metrics["n_total"], metrics["n_pos"], metrics["n_neg"]))
    print()
    print("{0:<14}{1:<11}{2:<24}{3:<10}".format("Metric", "Value", "95% CI", "SE"))
    print("-" * 70)

    acc_lo, acc_hi = metrics["accuracy_ci"]
    auc_lo, auc_hi = metrics["auc_ci"]
    mcc_lo, mcc_hi = metrics["mcc_ci"]

    print("{0:<14}{1:<11.4f}[{2:.4f} - {3:.4f}]  {4:<10}".format(
        "Accuracy", metrics["accuracy"], acc_lo, acc_hi, "N/A"))
    print("{0:<14}{1:<11.4f}[{2:.4f} - {3:.4f}]  {4:<10.4f}".format(
        "AUC", metrics["auc"], auc_lo, auc_hi, metrics["auc_se"]))
    print("{0:<14}{1:<11.4f}[{2:.4f} - {3:.4f}]  {4:<10}".format(
        "MCC", metrics["mcc"], mcc_lo, mcc_hi, "N/A"))
    print("{0:<14}{1:<11.4f}{2:<24}{3:<10}".format(
        "Precision", metrics["precision"], "N/A", "N/A"))
    print("{0:<14}{1:<11.4f}{2:<24}{3:<10}".format(
        "Sensitivity", metrics["sensitivity"], "N/A", "N/A"))
    print("{0:<14}{1:<11.4f}{2:<24}{3:<10}".format(
        "Specificity", metrics["specificity"], "N/A", "N/A"))
    print("{0:<14}{1:<11.4f}{2:<24}{3:<10}".format(
        "F1 score", metrics["f1"], "N/A", "N/A"))

    cm = metrics["confusion"]
    print()
    print("Confusion matrix: TN={0}, FP={1}, FN={2}, TP={3}".format(
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]))


# ============================================================
# Cross-validation
# ============================================================

def cross_validate(X_train, y_train):
    """Run stratified k-fold cross-validation on the training set."""
    print()
    print("Running {0}-fold stratified cross-validation...".format(CV_FOLDS))

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                         random_state=RANDOM_STATE)

    y_proba = cross_val_predict(
        build_model(), X_train, y_train, cv=cv,
        method="predict_proba", n_jobs=1,
    )[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return evaluate(y_train, y_pred, y_proba)


# ============================================================
# Visualization
# ============================================================

def plot_roc(metrics, save_path):
    """Save the ROC curve with the AUC and its 95% confidence interval."""
    auc_lo, auc_hi = metrics["auc_ci"]

    plt.figure(figsize=(8, 8))
    plt.plot(metrics["fpr"], metrics["tpr"], color="#1f77b4", lw=2.5,
             label="Hybrid Model (AUC = {0:.3f}, 95% CI: {1:.3f}-{2:.3f})".format(
                 metrics["auc"], auc_lo, auc_hi))
    plt.plot([0, 1], [0, 1], color="red", linestyle="--", lw=2,
             label="Random Classifier")
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=13,
               fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13,
               fontweight="bold")
    plt.title("ROC Curve with 95% Confidence Interval", fontsize=14,
              fontweight="bold")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, linestyle="--", alpha=0.3, linewidth=0.8)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(save_path.replace(".png", ".pdf"),
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("ROC curve saved to: {0}".format(save_path))


# ============================================================
# Prediction on new sequences
# ============================================================

def predict(sequences, model_path=MODEL_PATH):
    """
    Predict smORF probability for new DNA sequences.

    Parameters
    ----------
    sequences : list of str
    model_path : str

    Returns
    -------
    predictions : np.ndarray of int (0 or 1)
    probabilities : np.ndarray of float
    """
    package = joblib.load(model_path)
    rf = package["rf"]
    gb = package["gb"]
    vectorizer = package["vectorizer"]

    cleaned = [clean_sequence(s) for s in sequences]
    X = vectorizer.transform(cleaned)
    return predict_soft(rf, gb, X)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Hybrid smORF Prediction Model")
    print("=" * 70)
    print("Started: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    print("\nLoading dataset...")
    X_text, y = load_data()
    print("  Total sequences: {0}".format(len(X_text)))
    print("  Positive class:  {0}".format(int((y == 1).sum())))
    print("  Negative class:  {0}".format(int((y == 0).sum())))

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y,
    )
    print("\nTrain: {0}  |  Test: {1}".format(len(X_train_txt), len(X_test_txt)))

    print("\nExtracting {0}-mer features (min_df={1})...".format(KMER_SIZE, MIN_DF))
    X_train, X_test, vectorizer = vectorize(X_train_txt, X_test_txt)
    print("  Training matrix: {0}".format(X_train.shape))
    print("  Test matrix:     {0}".format(X_test.shape))
    print("  Vocabulary size: {0}".format(len(vectorizer.vocabulary_)))

    cv_metrics = cross_validate(X_train, y_train)
    print_metrics(cv_metrics, "{0}-fold cross-validation".format(CV_FOLDS))

    print("\nTraining final model on the full training set...")
    model = build_model()
    rf, gb = fit_model(model, X_train, y_train)

    y_pred, y_proba = predict_soft(rf, gb, X_test)
    holdout_metrics = evaluate(y_test, y_pred, y_proba)
    print_metrics(holdout_metrics, "Hold-out test set")

    print("\nSaving outputs...")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    joblib.dump({"rf": rf, "gb": gb, "vectorizer": vectorizer}, MODEL_PATH)
    print("  Model: {0}".format(MODEL_PATH))

    plot_roc(holdout_metrics, os.path.join(RESULTS_DIR, "roc_curve.png"))

    summary = pd.DataFrame([
        {
            "Evaluation": "Cross-validation",
            "Accuracy": cv_metrics["accuracy"],
            "Precision": cv_metrics["precision"],
            "Sensitivity": cv_metrics["sensitivity"],
            "Specificity": cv_metrics["specificity"],
            "F1": cv_metrics["f1"],
            "MCC": cv_metrics["mcc"],
            "AUC": cv_metrics["auc"],
        },
        {
            "Evaluation": "Hold-out test",
            "Accuracy": holdout_metrics["accuracy"],
            "Accuracy_CI_low": holdout_metrics["accuracy_ci"][0],
            "Accuracy_CI_high": holdout_metrics["accuracy_ci"][1],
            "Precision": holdout_metrics["precision"],
            "Sensitivity": holdout_metrics["sensitivity"],
            "Specificity": holdout_metrics["specificity"],
            "F1": holdout_metrics["f1"],
            "MCC": holdout_metrics["mcc"],
            "MCC_CI_low": holdout_metrics["mcc_ci"][0],
            "MCC_CI_high": holdout_metrics["mcc_ci"][1],
            "AUC": holdout_metrics["auc"],
            "AUC_CI_low": holdout_metrics["auc_ci"][0],
            "AUC_CI_high": holdout_metrics["auc_ci"][1],
        },
    ])
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print("  Metrics: {0}".format(summary_path))

    print("\nCompleted: {0}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    print("=" * 70)

    return model, vectorizer, holdout_metrics, cv_metrics


if __name__ == "__main__":
    main()
