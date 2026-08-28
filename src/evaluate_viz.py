import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import BASE_DIR
from src.model import ISLGestureClassifier
from src.train import load_dataset

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_evaluation_visualizations():
    """
    Evaluates trained GRU gesture classifier on full dataset,
    generates Confusion Matrix heatmap and performance visualization graphics.
    """
    print("\n=======================================================")
    print("  Generating ISL Model Evaluation Visualizations")
    print("=======================================================\n")

    classifier = ISLGestureClassifier()
    if not os.path.exists(classifier.model_path):
        print("Error: No trained model checkpoint found.")
        return

    X_data, y_data, class_names = load_dataset()

    y_preds = []
    confidences = []

    for i in range(len(X_data)):
        pred_label, conf, _ = classifier.predict(X_data[i])
        pred_idx = classifier.label_to_idx.get(pred_label, 0)
        y_preds.append(pred_idx)
        confidences.append(conf)

    y_preds = np.array(y_preds)
    confidences = np.array(confidences)

    cm = confusion_matrix(y_data, y_preds)

    # 1. Plot Confusion Matrix Heatmap using matplotlib
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Indian Sign Language (ISL) - Confusion Matrix',
           ylabel='Ground Truth Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, fontweight='bold')

    fig.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved Confusion Matrix Plot -> {cm_path}")

    # 2. Plot Per-Class Accuracy & Confidence Bar Chart
    per_class_acc = []
    per_class_conf = []

    for idx, cname in enumerate(class_names):
        mask = (y_data == idx)
        acc = np.mean(y_preds[mask] == y_data[mask]) * 100 if np.sum(mask) > 0 else 0
        avg_c = np.mean(confidences[mask]) * 100 if np.sum(mask) > 0 else 0
        per_class_acc.append(acc)
        per_class_conf.append(avg_c)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    x = np.arange(len(class_names))
    width = 0.35

    rects1 = ax.bar(x - width/2, per_class_acc, width, label='Validation Accuracy (%)', color='#2b5c8f')
    rects2 = ax.bar(x + width/2, per_class_conf, width, label='Avg Prediction Conf (%)', color='#4ba3e3')

    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('ISL Gesture Recognition Metrics by Category', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=15, ha='right')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Label bar values
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    perf_path = os.path.join(REPORTS_DIR, "performance_metrics.png")
    plt.savefig(perf_path)
    plt.close()
    print(f"Saved Performance Metrics Plot -> {perf_path}")

    print("\nVisualizations successfully compiled.")

if __name__ == "__main__":
    generate_evaluation_visualizations()
