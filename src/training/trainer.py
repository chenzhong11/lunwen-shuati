"""WDCNN-PHM 训练器模块。

基于 PyTorch 实现 WDCNN 模型的训练器，支持：
- Adam 优化器（论文明确指定）
- CrossEntropyLoss 损失函数（论文明确指定）
- 训练历史记录（loss, accuracy）
- 检查点保存与加载
- GPU/CPU 自动检测
- 训练时间记录
"""

import time
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)


class Trainer:
    """WDCNN 模型训练器。
    
    实现完整的训练循环，支持源域有标签数据训练，
    记录训练历史（loss、accuracy）和训练时间。
    
    Attributes:
        model: WDCNN 模型实例。
        config: 训练配置字典。
        device: 计算设备（自动检测 GPU/CPU）。
        optimizer: Adam 优化器。
        criterion: 交叉熵损失函数。
        history: 训练历史记录。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        """初始化训练器。
        
        Args:
            model: WDCNN 模型实例。
            config: 训练配置字典，可包含以下键：
                - lr (float): 学习率，默认 0.001。
                - weight_decay (float): L2 正则化系数，默认 1e-4。
                - betas (tuple): Adam 优化器的动量参数，默认 (0.9, 0.999)。
                - step_size (int): 学习率衰减步数，默认 20。
                - gamma (float): 学习率衰减因子，默认 0.5。
            device: 计算设备，为 None 时自动检测 GPU/CPU。
        """
        # GPU/CPU 自动检测
        if device is None:
            self.device = torch.device(
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device

        self.model = model.to(self.device)
        self.config = config

        lr = config.get("lr", 0.001)
        weight_decay = config.get("weight_decay", 1e-4)
        betas = config.get("betas", (0.9, 0.999))

        # Adam 优化器（论文明确指定）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

        # CrossEntropyLoss（论文明确指定）
        self.criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        step_size = config.get("step_size", 20)
        gamma = config.get("gamma", 0.5)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

        # 训练历史记录
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
            "epoch_time": [],
        }

        # 训练计时
        self.total_training_time: float = 0.0
        self.best_val_acc: float = 0.0
        self.best_model_state: Optional[Dict] = None

        logger.info(
            f"训练器初始化完成 | 设备: {self.device} | "
            f"学习率: {lr} | 权重衰减: {weight_decay}"
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """执行完整训练循环。
        
        Args:
            train_loader: 源域训练数据加载器（有标签数据）。
            val_loader: 目标域验证数据加载器（AdaBN 后），可选。
            epochs: 训练轮数，默认 100。
            
        Returns:
            训练历史字典，包含各指标列表。
        """
        logger.info(f"开始训练 | 总轮数: {epochs} | 训练样本数: {len(train_loader.dataset)}")
        if val_loader is not None:
            logger.info(f"验证样本数: {len(val_loader.dataset)}")

        overall_start = time.time()

        for epoch in range(1, epochs + 1):
            # 单 epoch 训练
            train_metrics = self.train_epoch(train_loader)

            # 验证（如果提供验证集）
            val_metrics: Optional[Dict[str, float]] = None
            if val_loader is not None:
                val_metrics = self.validate(val_loader)

            # 更新学习率调度器
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # 记录历史
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["lr"].append(current_lr)
            self.history["epoch_time"].append(train_metrics["epoch_time"])

            if val_metrics is not None:
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

                # 保存最佳模型
                if val_metrics["accuracy"] > self.best_val_acc:
                    self.best_val_acc = val_metrics["accuracy"]
                    self.best_model_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
            else:
                # 无验证集时，保存最后的模型
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # 日志输出
            log_msg = (
                f"Epoch [{epoch:3d}/{epochs}] | "
                f"训练损失: {train_metrics['loss']:.4f} | "
                f"训练准确率: {train_metrics['accuracy']:.4f} | "
                f"耗时: {train_metrics['epoch_time']:.2f}s"
            )
            if val_metrics is not None:
                log_msg += (
                    f" | 验证损失: {val_metrics['loss']:.4f} | "
                    f"验证准确率: {val_metrics['accuracy']:.4f}"
                )
            logger.info(log_msg)

        self.total_training_time = time.time() - overall_start
        logger.info(
            f"训练完成 | 总耗时: {self.total_training_time:.2f}s | "
            f"最佳验证准确率: {self.best_val_acc:.4f}"
        )

        # 加载最佳模型参数
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return self.history

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """单 epoch 训练。
        
        Args:
            train_loader: 源域训练数据加载器（有标签数据）。
            
        Returns:
            包含 loss、accuracy、epoch_time 的指标字典。
        """
        self.model.train()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).long()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 反向传播与优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 统计
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / total if total > 0 else 0.0
        epoch_acc = correct / total if total > 0 else 0.0

        return {
            "loss": epoch_loss,
            "accuracy": epoch_acc,
            "epoch_time": epoch_time,
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证模型。
        
        Args:
            val_loader: 验证数据加载器。
            
        Returns:
            包含 loss 和 accuracy 的指标字典。
        """
        self.model.eval()

        running_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).long()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total if total > 0 else 0.0
        val_acc = correct / total if total > 0 else 0.0

        return {"loss": val_loss, "accuracy": val_acc}

    def save_checkpoint(self, path: str) -> None:
        """保存训练检查点。
        
        Args:
            path: 检查点文件保存路径。
        """
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "total_training_time": self.total_training_time,
            "config": self.config,
        }

        torch.save(checkpoint, str(checkpoint_path))
        logger.info(f"检查点已保存至: {checkpoint_path}")

    def load_checkpoint(self, path: str) -> None:
        """加载训练检查点。
        
        Args:
            path: 检查点文件路径。
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.total_training_time = checkpoint.get("total_training_time", 0.0)

        logger.info(f"检查点已加载: {checkpoint_path} | 最佳验证准确率: {self.best_val_acc:.4f}")
