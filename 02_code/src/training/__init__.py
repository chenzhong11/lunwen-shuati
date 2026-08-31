"""WDCNN-PHM 训练与评估模块。

提供模型训练 (Trainer) 和评估 (Evaluator) 功能，
用于源域有标签数据训练、AdaBN 后目标域评估的迁移学习流程。
"""

from .trainer import Trainer
from .evaluator import Evaluator

__all__ = ["Trainer", "Evaluator"]
