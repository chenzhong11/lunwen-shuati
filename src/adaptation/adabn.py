"""
AdaBN (Adaptive Batch Normalization) 域适应模块

本模块实现了 AdaBN 算法，用于无监督域适应。
AdaBN 通过替换 BatchNorm 层的统计量来适应目标域分布。

核心思想：
- 源域和目标域的差异主要体现在特征分布的均值和方差上
- BatchNorm 层的 running_mean 和 running_var 编码了源域的特征分布
- 通过用目标域统计量替换这些统计量，可以实现域适应

论文引用：
- Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016 Workshop
- 算法细节见论文 Algorithm 1

关键约束（论文 Algorithm 1 明确）：
1. 目标域数据无标签（unsupervised）
2. γ, β 保持源域训练值不变
3. 统计量使用目标域所有样本计算（非 mini-batch）
4. AdaBN 发生在测试前
"""

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base_adapter import BaseAdapter


class AdaBN(BaseAdapter):
    """
    AdaBN (Adaptive Batch Normalization) 域适应实现
    
    严格按照论文 Algorithm 1 实现的 AdaBN 算法。
    用于将源域训练的模型适应到目标域，无需目标域标签。
    
    论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016
    
    Attributes:
        model: 待适应的神经网络模型
        source_stats: 源域 BatchNorm 层的统计量
        target_stats: 目标域 BatchNorm 层的统计量
        is_adapted: 是否已完成适应
    """
    
    def __init__(self, model: nn.Module) -> None:
        """
        初始化 AdaBN 适配器
        
        Args:
            model: 待适应的神经网络模型，应具有 get_bn_layers() 方法
        """
        super().__init__(model)
        self.source_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self.target_stats: Dict[str, Dict[str, torch.Tensor]] = {}
        self._hooks: List[Any] = []
        self._activations: Dict[str, List[torch.Tensor]] = {}
    
    def collect_source_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        收集源域训练后的 BN 统计量
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016, Section 3.1
        
        在源域训练完成后调用，收集模型在源域上学到的 BatchNorm 统计量。
        包括：
        - running_mean: 运行均值（μ_s）
        - running_var: 运行方差（σ_s²）
        - gamma: 缩放参数（γ）
        - beta: 偏移参数（β）
        
        Returns:
            包含源域统计量的字典，结构为 {layer_name: {'running_mean': tensor, 'running_var': tensor, 'gamma': tensor, 'beta': tensor}}
        """
        self.source_stats = {}
        
        # 遍历所有 BatchNorm 层
        bn_layers = self.model.get_bn_layers()
        
        for i, bn_layer in enumerate(bn_layers):
            layer_name = f'bn_layer_{i}'
            
            # 收集统计量
            self.source_stats[layer_name] = {
                'running_mean': bn_layer.running_mean.clone().detach(),
                'running_var': bn_layer.running_var.clone().detach(),
                'gamma': bn_layer.weight.clone().detach(),  # 缩放参数
                'beta': bn_layer.bias.clone().detach(),     # 偏移参数
            }
        
        return self.source_stats
    
    def compute_target_stats(self, target_loader: DataLoader) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        计算目标域的统计量
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016, Algorithm 1, Step 4-7
        
        按照论文 Algorithm 1 实现：
        1. 切换到 eval mode
        2. 遍历目标域所有样本（无标签）
        3. 收集每个 BN 层的激活值
        4. 计算 μ_t = E[x_t], σ_t² = Var[x_t]
        
        关键约束：论文明确使用目标域所有样本计算统计量（非 mini-batch）
        
        Args:
            target_loader: 目标域数据加载器，数据应为无标签的（仅特征）
            
        Returns:
            包含目标域统计量的字典，结构为 {layer_name: {'mean': tensor, 'var': tensor}}
        """
        # 切换到 eval mode（论文 Algorithm 1, Step 3）
        self.model.eval()
        
        # 初始化激活值收集器
        self._activations = {}
        bn_layers = self.model.get_bn_layers()
        
        for i in range(len(bn_layers)):
            layer_name = f'bn_layer_{i}'
            self._activations[layer_name] = []
        
        # 注册钩子收集激活值
        self._hooks = []
        for i, bn_layer in enumerate(bn_layers):
            layer_name = f'bn_layer_{i}'
            hook = bn_layer.register_forward_hook(self._create_hook(layer_name))
            self._hooks.append(hook)
        
        # 遍历目标域所有样本（论文 Algorithm 1, Step 4）
        with torch.no_grad():
            for batch_data in target_loader:
                # 处理不同的数据格式
                if isinstance(batch_data, (list, tuple)):
                    # 如果是 (data, label) 格式，只取 data
                    inputs = batch_data[0]
                else:
                    inputs = batch_data
                
                # 确保输入是张量
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs)
                
                # 前向传播收集激活值
                self.model(inputs)
        
        # 移除钩子
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
        # 计算目标域统计量（论文 Algorithm 1, Step 5-7）
        self.target_stats = {}
        
        for layer_name, activations in self._activations.items():
            if len(activations) == 0:
                continue
            
            # 拼接所有激活值
            all_activations = torch.cat(activations, dim=0)
            
            # 计算均值 μ_t = E[x_t]（论文 Algorithm 1, Step 5）
            mean_t = all_activations.mean(dim=0)
            
            # 计算方差 σ_t² = Var[x_t]（论文 Algorithm 1, Step 6）
            var_t = all_activations.var(dim=0, unbiased=False)
            
            self.target_stats[layer_name] = {
                'mean': mean_t,
                'var': var_t,
            }
        
        return self.target_stats
    
    def _create_hook(self, layer_name: str):
        """
        创建前向传播钩子，用于收集 BatchNorm 层的激活值
        
        Args:
            layer_name: 层名称
            
        Returns:
            钩子函数
        """
        def hook(module, input, output):
            # 收集输入激活值（BatchNorm 的输入）
            if layer_name not in self._activations:
                self._activations[layer_name] = []
            self._activations[layer_name].append(input[0].detach().cpu())
        
        return hook
    
    def apply_adaptation(self) -> None:
        """
        应用域适应
        
        论文引用：Li et al., "Revisiting Batch Normalization for Practical Domain Adaptation", ICLR 2016, Algorithm 1, Step 8-9
        
        按照论文 Algorithm 1 实现：
        1. 用 μ_t, σ_t² 替换 running_mean, running_var（论文 Algorithm 1, Step 8）
        2. 保持 γ, β 不变（论文 Algorithm 1, Step 9，论文明确指出）
        
        Raises:
            RuntimeError: 如果在调用此方法前未完成 collect_source_stats() 和 compute_target_stats()
        """
        # 检查是否已收集统计量
        if not self.source_stats:
            raise RuntimeError("请先调用 collect_source_stats() 收集源域统计量")
        if not self.target_stats:
            raise RuntimeError("请先调用 compute_target_stats() 计算目标域统计量")
        
        # 遍历所有 BatchNorm 层
        bn_layers = self.model.get_bn_layers()
        
        for i, bn_layer in enumerate(bn_layers):
            layer_name = f'bn_layer_{i}'
            
            if layer_name not in self.target_stats:
                continue
            
            # 用目标域统计量替换 running_mean 和 running_var（论文 Algorithm 1, Step 8）
            # μ_t 替换 running_mean
            bn_layer.running_mean = self.target_stats[layer_name]['mean'].clone()
            # σ_t² 替换 running_var
            bn_layer.running_var = self.target_stats[layer_name]['var'].clone()
            
            # 保持 γ, β 不变（论文 Algorithm 1, Step 9）
            # 无需修改，因为 bn_layer.weight (gamma) 和 bn_layer.bias (beta) 已经是源域训练值
        
        self.is_adapted = True
    
    def evaluate(self, target_loader: DataLoader) -> Dict[str, float]:
        """
        使用适应后的模型在目标域上评估
        
        在完成域适应后，使用适应后的模型在目标域数据上进行评估。
        评估指标包括准确率和损失。
        
        Args:
            target_loader: 目标域数据加载器（应包含标签用于评估）
            
        Returns:
            包含评估指标的字典，如 {'accuracy': 0.95, 'loss': 0.1}
            
        Raises:
            RuntimeError: 如果在调用此方法前未完成 apply_adaptation()
        """
        if not self.is_adapted:
            raise RuntimeError("请先调用 apply_adaptation() 应用域适应")
        
        # 切换到 eval mode
        self.model.eval()
        
        correct = 0
        total = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch_data in target_loader:
                # 处理不同的数据格式
                if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                    inputs, labels = batch_data[0], batch_data[1]
                else:
                    # 如果没有标签，无法计算准确率
                    continue
                
                # 确保输入是张量
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.tensor(inputs)
                if not isinstance(labels, torch.Tensor):
                    labels = torch.tensor(labels)
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        accuracy = correct / total
        avg_loss = total_loss / total
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
        }
    
    def get_adaptation_stats(self) -> Dict[str, Any]:
        """
        返回适应前后的统计量（用于审计）
        
        提供完整的适应过程审计信息，包括源域统计量、目标域统计量，
        以及两者之间的差异分析。
        
        Returns:
            包含适应统计量的字典，结构为：
            {
                'source_stats': 源域统计量,
                'target_stats': 目标域统计量,
                'adaptation_diff': 统计量差异分析,
                'is_adapted': 是否已完成适应
            }
        """
        adaptation_diff = {}
        
        # 计算统计量差异
        for layer_name in self.source_stats:
            if layer_name in self.target_stats:
                source_mean = self.source_stats[layer_name]['running_mean']
                target_mean = self.target_stats[layer_name]['mean']
                source_var = self.source_stats[layer_name]['running_var']
                target_var = self.target_stats[layer_name]['var']
                
                # 计算均值差异
                mean_diff = torch.abs(target_mean - source_mean).mean().item()
                # 计算方差差异
                var_diff = torch.abs(target_var - source_var).mean().item()
                
                adaptation_diff[layer_name] = {
                    'mean_diff': mean_diff,
                    'var_diff': var_diff,
                }
        
        return {
            'source_stats': self.source_stats,
            'target_stats': self.target_stats,
            'adaptation_diff': adaptation_diff,
            'is_adapted': self.is_adapted,
        }
