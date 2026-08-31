"""WDCNN-PHM 评估器模块。

基于 PyTorch 实现 WDCNN 模型的评估器，支持：
- 目标域数据评估（AdaBN 后）
- 混淆矩阵计算
- 每类准确率统计
- JSON 格式评估报告生成
- GPU/CPU 自动检测
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Evaluator:
    """WDCNN 模型评估器。
    
    用于目标域数据（AdaBN 后）的模型评估，
    提供准确率、混淆矩阵、每类准确率等详细评估指标。
    
    Attributes:
        model: WDCNN 模型实例。
        device: 计算设备（自动检测 GPU/CPU）。
        class_names: 类别名称列表（可选）。
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        """初始化评估器。
        
        Args:
            model: WDCNN 模型实例。
            device: 计算设备，为 None 时自动检测 GPU/CPU。
            class_names: 类别名称列表，用于报告可读性。
        """
        # GPU/CPU 自动检测
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names

        logger.info(f"评估器初始化完成 | 设备: {self.device}")

    @torch.no_grad()
    def evaluate(
        self, data_loader: DataLoader
    ) -> Dict[str, Any]:
        """评估模型。
        
        在目标域数据（AdaBN 后）上评估模型性能，
        返回整体准确率、混淆矩阵和每类准确率。
        
        Args:
            data_loader: 目标域测试数据加载器。
            
        Returns:
            包含以下键的评估结果字典：
            - accuracy (float): 整体准确率。
            - confusion_matrix (np.ndarray): 混淆矩阵。
            - per_class_accuracy (Dict[str, float]): 每类准确率。
            - total_samples (int): 总样本数。
            - correct_samples (int): 正确预测样本数。
            - all_predictions (np.ndarray): 所有预测结果。
            - all_targets (np.ndarray): 所有真实标签。
        """
        logger.info(f"开始评估 | 测试样本数: {len(data_loader.dataset)}")

        all_predictions: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        eval_start = time.time()

        for inputs, targets in data_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            outputs = self.model(inputs)
            _, predicted = outputs.max(1)

            all_predictions.append(predicted.cpu())
            all_targets.append(targets.cpu())

        eval_time = time.time() - eval_start

        # 拼接所有预测和目标
        predictions = torch.cat(all_predictions).numpy()
        targets = torch.cat(all_targets).numpy()

        # 计算混淆矩阵
        confusion_mat = self.compute_confusion_matrix(targets, predictions)

        # 计算准确率
        correct = (predictions == targets).sum()
        total = len(targets)
        accuracy = correct / total if total > 0 else 0.0

        # 计算每类准确率
        per_class_acc = self._compute_per_class_accuracy(confusion_mat)

        logger.info(
            f"评估完成 | 准确率: {accuracy:.4f} | "
            f"正确: {correct}/{total} | 耗时: {eval_time:.2f}s"
        )

        return {
            "accuracy": float(accuracy),
            "confusion_matrix": confusion_mat,
            "per_class_accuracy": per_class_acc,
            "total_samples": int(total),
            "correct_samples": int(correct),
            "all_predictions": predictions,
            "all_targets": targets,
            "eval_time": eval_time,
        }

    def compute_confusion_matrix(
        self, targets: np.ndarray, predictions: np.ndarray
    ) -> np.ndarray:
        """计算混淆矩阵。
        
        Args:
            targets: 真实标签数组。
            predictions: 预测标签数组。
            
        Returns:
            混淆矩阵，形状为 (num_classes, num_classes)。
        """
        num_classes = int(max(targets.max(), predictions.max())) + 1
        confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

        for t, p in zip(targets, predictions):
            confusion_mat[int(t)][int(p)] += 1

        return confusion_mat

    def _compute_per_class_accuracy(
        self, confusion_mat: np.ndarray
    ) -> Dict[str, float]:
        """从混淆矩阵计算每类准确率。
        
        Args:
            confusion_mat: 混淆矩阵。
            
        Returns:
            每类准确率字典，键为类别名称或索引。
        """
        num_classes = confusion_mat.shape[0]
        per_class_acc: Dict[str, float] = {}

        for i in range(num_classes):
            class_total = confusion_mat[i].sum()
            if class_total > 0:
                acc = float(confusion_mat[i][i] / class_total)
            else:
                acc = 0.0

            # 使用类别名称（如果提供）
            class_name = (
                self.class_names[i]
                if self.class_names and i < len(self.class_names)
                else f"类别{i}"
            )
            per_class_acc[class_name] = acc

        return per_class_acc

    def generate_report(
        self,
        data_loader: DataLoader,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """生成评估报告。
        
        在目标域数据上评估模型并生成 JSON 格式的评估报告。
        
        Args:
            data_loader: 目标域测试数据加载器。
            save_path: 报告保存路径（可选）。
            
        Returns:
            完整的评估报告字典（JSON 可序列化）。
        """
        # 执行评估
        results = self.evaluate(data_loader)

        # 构建报告
        report: Dict[str, Any] = {
            "model": self.model.__class__.__name__,
            "device": str(self.device),
            "evaluation_domain": "target_domain_after_adabn",
            "overall_metrics": {
                "accuracy": results["accuracy"],
                "total_samples": results["total_samples"],
                "correct_samples": results["correct_samples"],
                "eval_time_seconds": round(results["eval_time"], 2),
            },
            "per_class_accuracy": results["per_class_accuracy"],
            "confusion_matrix": results["confusion_matrix"].tolist(),
        }

        # 保存报告
        if save_path is not None:
            save_file = Path(save_path)
            save_file.parent.mkdir(parents=True, exist_ok=True)

            with open(save_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info(f"评估报告已保存至: {save_file}")

        return report
