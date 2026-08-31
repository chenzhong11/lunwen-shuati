"""
数据模块
========

提供 CWRU 数据加载、预处理和场景创建功能。
"""

from .cwru_loader import (
    CWRULoader,
    create_scenario,
    segment_signal,
    normalize_zscore,
    CWRU_FILE_MAP,
    FAULT_LABEL_MAP,
)
from .leakage_checker import DataLeakageChecker

__all__ = [
    'CWRULoader',
    'create_scenario',
    'segment_signal',
    'normalize_zscore',
    'CWRU_FILE_MAP',
    'FAULT_LABEL_MAP',
    'DataLeakageChecker',
]
