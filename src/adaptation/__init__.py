"""
域适应模块

本模块提供了域适应算法的实现，用于将源域训练的模型适应到目标域。
目前实现了 AdaBN (Adaptive Batch Normalization) 算法。

主要功能：
- BaseAdapter: 域适应适配器的抽象基类
- AdaBN: AdaBN 算法实现，用于无监督域适应

论文引用：
- Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016 Workshop
"""

from .base_adapter import BaseAdapter
from .adabn import AdaBN

__all__ = ['BaseAdapter', 'AdaBN']

__version__ = '0.1.0'
