"""随机种子管理模块 - 确保实验可重复性"""
import random
import numpy as np
import os
from typing import Optional

def set_seed(seed: int = 42, deterministic: bool = True, benchmark: bool = False) -> dict:
    """
    设置全局随机种子
    
    Args:
        seed: 随机种子值
        deterministic: 是否启用确定性模式
        benchmark: 是否启用 cudnn benchmark
    
    Returns:
        包含种子信息的字典
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = benchmark
    except ImportError:
        pass
    
    return {"seed": seed, "deterministic": deterministic, "benchmark": benchmark}
