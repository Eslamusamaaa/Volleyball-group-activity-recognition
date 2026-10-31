"""Accuracy, confusion matrix, per-class precision/recall/F1 (Section 9)."""
import numpy as np


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def per_class_prf1(y_true, y_pred, num_classes, class_names=None):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    rows = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        name = class_names[c] if class_names else str(c)
        rows.append({"class": name, "precision": precision, "recall": recall, "f1": f1, "support": int(cm[c, :].sum())})
    return rows, cm


def format_report(y_true, y_pred, num_classes, class_names):
    rows, cm = per_class_prf1(y_true, y_pred, num_classes, class_names)
    lines = [f"accuracy: {accuracy(y_true, y_pred):.4f}", ""]
    lines.append(f"{'class':12s} {'precision':>9s} {'recall':>7s} {'f1':>7s} {'support':>7s}")
    for r in rows:
        lines.append(f"{r['class']:12s} {r['precision']:9.3f} {r['recall']:7.3f} {r['f1']:7.3f} {r['support']:7d}")
    lines.append("\nconfusion matrix (rows=true, cols=pred):")
    lines.append("  " + " ".join(f"{n[:6]:>6s}" for n in class_names))
    for i, n in enumerate(class_names):
        lines.append(f"{n[:6]:>6s} " + " ".join(f"{v:6d}" for v in cm[i]))
    return "\n".join(lines)
