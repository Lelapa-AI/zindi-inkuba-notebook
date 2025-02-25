import csv
from collections import Counter

import numpy as np


def evaluate_zindi(csv_file_path):
    with open(csv_file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        scores = []
        y_pred_sent = []
        y_true_sent = []
        y_pred_xnli = []
        y_true_xnli = []
        chrfs_scores = []

        for row in reader:
            if "sent" in row["ID"] or "xnli" in row["ID"]:
                if "sent" in row["ID"] and "swahili" in row["ID"]:
                    labels = ["Chanya", "Wastani", "Hasi"]
                if "sent" in row["ID"] and "hausa" in row["ID"]:
                    labels = ["Kyakkyawa", "Tsaka-tsaki", "Korau"]
                if "xnli" in row["ID"]:
                    labels = ["0", "1", "2"]

                # Use the output of process_likelihood directly
                predicted_label = int(row["Response"])
                label_to_id = {label: i for i, label in enumerate(labels)}

                if "xnli" in row["ID"]:
                    y_pred_xnli.append(predicted_label)
                    y_true_xnli.append(label_to_id[row["Targets"]])
                if "sent" in row["ID"]:
                    y_pred_sent.append(predicted_label)
                    y_true_sent.append(label_to_id[row["Targets"]])

            elif "mt" in row["ID"]:
                chrf_pred = row["Response"]
                chrf_true = row["Targets"]
                chrfs = chrF(reference=chrf_true, hypothesis=chrf_pred)
                chrfs_scores.append(chrfs)

        # F1 score for sentiment
        f1_sent = calculate_f1(np.array(y_true_sent), np.array(y_pred_sent), 3)
        scores.append(f1_sent)
        # F1 score for xnli
        f1_xnli = calculate_f1(np.array(y_true_xnli), np.array(y_pred_xnli), 3)
        scores.append(f1_xnli)
        # chrF score for mt
        chrfs_score = np.mean(chrfs_scores)
        scores.append(chrfs_score)

        # Zindi score: Average of all performances
        zindi_score = np.mean(scores)

    # Round to 4 decimal places and multiply by 100
    zindi_score = round(zindi_score, 4)
    zindi_score *= 100

    return zindi_score


# From scratch implementation of chrf
def get_char_ngrams(sentence, n):
    """Generate character n-grams from a sentence."""
    sentence = sentence.replace(" ", "")  # Remove spaces for chrF
    return [sentence[i : i + n] for i in range(len(sentence) - n + 1)]


def precision_recall(reference, hypothesis, n):
    """Calculate precision and recall for character n-grams."""
    ref_ngrams = get_char_ngrams(reference, n)
    hyp_ngrams = get_char_ngrams(hypothesis, n)

    ref_count = Counter(ref_ngrams)
    hyp_count = Counter(hyp_ngrams)

    common_ngrams = ref_count & hyp_count
    true_positives = sum(common_ngrams.values())

    precision = true_positives / max(len(hyp_ngrams), 1)
    recall = true_positives / max(len(ref_ngrams), 1)

    return precision, recall


def f_score(precision, recall, beta=1):
    """Calculate the F1 score."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)


def chrF(reference, hypothesis, max_n=6, beta=2):
    """Calculate the chrF score from scratch."""
    precisions = []
    recalls = []

    for n in range(1, max_n + 1):
        precision, recall = precision_recall(reference, hypothesis, n)
        precisions.append(precision)
        recalls.append(recall)

    avg_precision = sum(precisions) / max_n
    avg_recall = sum(recalls) / max_n

    return f_score(avg_precision, avg_recall, beta)


# From scratch implementation f1score 3 class
def calculate_f1(true_labels, pred_labels, num_classes):
    f1_scores = []

    for i in range(num_classes):
        TP = np.sum((true_labels == i) & (pred_labels == i))  # True Positives
        FP = np.sum((true_labels != i) & (pred_labels == i))  # False Positives
        FN = np.sum((true_labels == i) & (pred_labels != i))  # False Negatives

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # Calculate F1 score
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        f1_scores.append(f1)

    macro_f1 = np.mean(f1_scores)

    return macro_f1
