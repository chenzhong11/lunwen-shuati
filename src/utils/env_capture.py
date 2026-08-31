"""环境快照模块 - 记录运行环境信息"""
import platform
import sys
from typing import Dict, Any

def capture_environment() -> Dict[str, Any]:
    """
    捕获当前运行环境信息
    
    Returns:
        包含环境信息的字典
    """
    env_info = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "os": platform.system(),
        "os_version": platform.version(),
    }
    
    try:
        import torch
        env_info["pytorch_version"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env_info["cuda_version"] = torch.version.cuda
            env_info["gpu_name"] = torch.cuda.get_device_name(0)
            env_info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        env_info["pytorch_version"] = "not installed"
        env_info["cuda_available"] = False
    
    try:
        import numpy as np
        env_info["numpy_version"] = np.__version__
    except ImportError:
        pass
    
    return env_info
