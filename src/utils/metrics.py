"""指标计算模块"""
import numpy as np
from typing import Dict, List, Optional, Tuple

def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """计算准确率"""
    return np.mean(predictions == labels)

def compute_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """计算混淆矩阵"""
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for pred, label in zip(predictions, labels):
        matrix[label][pred] += 1
    return matrix

def compute_per_class_accuracy(confusion_matrix: np.ndarray) -> Dict[int, float]:
    """计算每个类别的准确率"""
    per_class = {}
    for i in range(len(confusion_matrix)):
        total = confusion_matrix[i].sum()
        if total > 0:
            per_class[i] = confusion_matrix[i][i] / total
        else:
            per_class[i] = 0.0
    return per_class
