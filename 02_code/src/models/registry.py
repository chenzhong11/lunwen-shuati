"""
模型注册表模块

本模块提供了一个简单的模型注册表，支持按名称获取模型类。
便于配置化地选择和实例化模型。
"""

from typing import Dict, Type, Optional, Any

import torch.nn as nn

from .base_model import BaseModel


class ModelRegistry:
    """
    模型注册表
    
    提供统一的接口来注册和获取模型类。
    支持按名称查找模型，并可传递自定义参数进行实例化。
    
    Example:
        >>> registry = ModelRegistry()
        >>> registry.register("wdcnn", WDCNN)
        >>> model = registry.get("wdcnn", num_classes=10)
    """
    
    def __init__(self) -> None:
        """
        初始化模型注册表
        """
        self._registry: Dict[str, Type[BaseModel]] = {}
    
    def register(self, name: str, model_class: Type[BaseModel]) -> None:
        """
        注册模型类
        
        Args:
            name: 模型名称（不区分大小写）
            model_class: 模型类，必须继承自 BaseModel
            
        Raises:
            TypeError: 如果 model_class 不是 BaseModel 的子类
            ValueError: 如果 name 已被注册
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"model_class 必须是 BaseModel 的子类，"
                f"但得到了 {model_class.__name__}"
            )
        
        name_lower = name.lower()
        
        if name_lower in self._registry:
            raise ValueError(f"模型 '{name}' 已被注册")
        
        self._registry[name_lower] = model_class
    
    def get(self, name: str, **kwargs: Any) -> BaseModel:
        """
        获取并实例化模型
        
        Args:
            name: 模型名称（不区分大小写）
            **kwargs: 传递给模型构造函数的参数
            
        Returns:
            实例化后的模型
            
        Raises:
            KeyError: 如果模型名称未注册
        """
        name_lower = name.lower()
        
        if name_lower not in self._registry:
            raise KeyError(
                f"未找到模型 '{name}'。"
                f"可用的模型: {list(self._registry.keys())}"
            )
        
        model_class = self._registry[name_lower]
        return model_class(**kwargs)
    
    def list_models(self) -> list:
        """
        列出所有已注册的模型名称
        
        Returns:
            模型名称列表
        """
        return list(self._registry.keys())
    
    def is_registered(self, name: str) -> bool:
        """
        检查模型是否已注册
        
        Args:
            name: 模型名称
            
        Returns:
            如果已注册返回 True，否则返回 False
        """
        return name.lower() in self._registry


# 全局模型注册表实例
model_registry = ModelRegistry()


def register_model(name: str) -> callable:
    """
    模型注册装饰器
    
    用于将模型类注册到全局注册表。
    
    Args:
        name: 模型名称
        
    Returns:
        装饰器函数
        
    Example:
        >>> @register_model("wdcnn")
        ... class WDCNN(BaseModel):
        ...     pass
    """
    def decorator(cls: Type[BaseModel]) -> Type[BaseModel]:
        model_registry.register(name, cls)
        return cls
    return decorator
