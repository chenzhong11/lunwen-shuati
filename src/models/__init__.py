"""
WDCNN PHM 模型模块

本模块提供了用于预测与健康管理 (PHM) 的深度学习模型。
主要实现了 WDCNN (Wide Deep Convolutional Neural Network) 架构。

Example:
    >>> from src.models import WDCNN, BaseModel
    >>> model = WDCNN(num_classes=10)
    >>> print(model)
"""

from typing import List

# 导入基类
from .base_model import BaseModel

# 导入具体模型
from .wdcnn import WDCNN

# 导入注册表
from .registry import model_registry, register_model, ModelRegistry

# 导出列表
__all__: List[str] = [
    # 基类
    "BaseModel",
    
    # 模型实现
    "WDCNN",
    
    # 注册表
    "ModelRegistry",
    "model_registry",
    "register_model",
]

# 自动注册默认模型
try:
    model_registry.register("wdcnn", WDCNN)
except ValueError:
    # 如果已注册则忽略
    pass
