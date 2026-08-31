"""
WDCNN 模型验证脚本

使用方法:
    D:/Anaconda3/python.exe E:/projects/wdcnn-phm-reproduction/src/models/verify_model.py

验证内容:
1. 模型能否正常实例化
2. 前向传播输出 shape 是否正确
3. BN 层获取是否正常
"""

import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

def verify_model():
    """验证 WDCNN 模型"""
    try:
        import torch
        print(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        print("错误: 未安装 PyTorch。请先安装 PyTorch:")
        print("  使用清华镜像源: pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return False
    
    try:
        from src.models import WDCNN, BaseModel
        print("[PASS] 模块导入成功")
    except ImportError as e:
        print(f"[FAIL] 模块导入失败: {e}")
        return False
    
    # 测试 1: 模型实例化
    try:
        model = WDCNN(num_classes=10, input_length=2048)
        print("[PASS] 模型实例化成功")
    except Exception as e:
        print(f"[FAIL] 模型实例化失败: {e}")
        return False
    
    # 测试 2: 前向传播
    try:
        batch_size = 32
        x = torch.randn(batch_size, 1, 2048)
        output = model(x)
        
        expected_shape = (batch_size, 10)
        assert output.shape == expected_shape, \
            f"输出 shape 错误: 期望 {expected_shape}, 实际 {output.shape}"
        print(f"[PASS] 前向传播成功, 输出 shape: {output.shape}")
    except Exception as e:
        print(f"[FAIL] 前向传播失败: {e}")
        return False
    
    # 测试 3: BN 层获取
    try:
        bn_layers = model.get_bn_layers()
        assert len(bn_layers) == 6, \
            f"BN 层数量错误: 期望 6, 实际 {len(bn_layers)}"
        print(f"[PASS] BN 层获取成功, 共 {len(bn_layers)} 层")
    except Exception as e:
        print(f"[FAIL] BN 层获取失败: {e}")
        return False
    
    # 测试 4: 模型注册表
    try:
        from src.models import model_registry
        assert model_registry.is_registered("wdcnn"), "WDCNN 未注册"
        model2 = model_registry.get("wdcnn", num_classes=10, input_length=2048)
        print("[PASS] 模型注册表工作正常")
    except Exception as e:
        print(f"[FAIL] 模型注册表测试失败: {e}")
        return False
    
    # 测试 5: 继承关系
    try:
        assert isinstance(model, BaseModel), "WDCNN 应该是 BaseModel 的实例"
        print("[PASS] 继承关系正确")
    except Exception as e:
        print(f"[FAIL] 继承关系测试失败: {e}")
        return False
    
    # 打印模型摘要
    print("\n" + "="*50)
    print("模型结构摘要:")
    print("="*50)
    print(f"输入形状: (batch, 1, 2048)")
    print(f"输出形状: (batch, {model.num_classes})")
    print(f"特征长度: {model._feature_length}")
    print(f"BN 层数量: {len(model.get_bn_layers())}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    print("\n[PASS] 所有已执行模型检查通过!")
    return True


if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)
