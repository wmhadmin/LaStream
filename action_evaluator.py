import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
from typing import List, Tuple, Optional


class ActionEvaluator(nn.Module):
    """
    用于评估并行度动作质量的前馈神经网络。
    结合GCN应用嵌入和并行度配置来预测动作价值。
    """
     
    def __init__(self,
                 app_embedding_dim: int = 64,
                 max_parallelism: int = 16,
                 hidden_dims: List[int] = [256, 512, 256, 128],
                 dropout: float = 0.2,
                 max_operators: int = 10):
        """
        初始化动作评估网络。
        
        Args:
            app_embedding_dim: GCN应用嵌入的维度
            max_parallelism: 用于标准化的最大并行度
            hidden_dims: 隐藏层维度列表
            dropout: Dropout概率
            max_operators: 最大操作符数量（用于确定网络输入维度）
        """
        super(ActionEvaluator, self).__init__()
        
        self.app_embedding_dim = app_embedding_dim
        self.max_parallelism = max_parallelism
        self.dropout = dropout
        self.max_operators = max_operators
        
        # 输入维度 = 应用嵌入维度 + 并行度向量维度
        # 现在可以确定并行度向量的大小
        input_dim = app_embedding_dim + max_operators
        
        # 立即构建网络层
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            # 为除输出层外的所有层添加批标准化
            if i < len(dims) - 2:
                self.batch_norms.append(nn.BatchNorm1d(dims[i + 1]))
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # 模型加载状态
        self._model_loaded = False
        self._loaded_model_path = None
        
    def _pad_parallelism_config(self, parallelism_config: torch.Tensor) -> torch.Tensor:
        """
        将并行度配置填充到固定长度。
        
        Args:
            parallelism_config: 并行度配置张量 [batch_size, num_operators]
            
        Returns:
            torch.Tensor: 填充后的并行度配置 [batch_size, max_operators]
        """
        batch_size, num_operators = parallelism_config.shape
        
        if num_operators > self.max_operators:
            # 如果操作符数量超过最大值，截断
            return parallelism_config[:, :self.max_operators]
        elif num_operators < self.max_operators:
            # 如果操作符数量少于最大值，用0填充
            padding = torch.zeros(batch_size, self.max_operators - num_operators, 
                                device=parallelism_config.device, dtype=parallelism_config.dtype)
            return torch.cat([parallelism_config, padding], dim=1)
        else:
            return parallelism_config
    
    def forward(self, 
                app_embedding: torch.Tensor,
                parallelism_config: torch.Tensor) -> torch.Tensor:
        """
        前向传播评估动作质量。
        
        Args:
            app_embedding: GCN的应用嵌入 [batch_size, app_embedding_dim]
            parallelism_config: 并行度配置 [batch_size, num_operators]
            
        Returns:
            torch.Tensor: 动作价值预测 [batch_size, 1]
        """
        # 处理单样本情况
        if app_embedding.dim() == 1:
            app_embedding = app_embedding.unsqueeze(0)
        if parallelism_config.dim() == 1:
            parallelism_config = parallelism_config.unsqueeze(0)
        
        batch_size = app_embedding.size(0)
        
        # 将并行度配置填充到固定长度
        padded_parallelism = self._pad_parallelism_config(parallelism_config)
        
        # 将并行度配置标准化到 [0, 1]
        normalized_parallelism = padded_parallelism.float() / self.max_parallelism
        
        # 连接嵌入
        combined_input = torch.cat([app_embedding, normalized_parallelism], dim=1)
        
        # 网络前向传播
        x = combined_input
        
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            # 如果批量大小大于1且存在批标准化层，则应用批标准化
            if self.batch_norms and batch_size > 1:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout_layer(x)
        
        # 输出层（回归任务无激活函数）
        x = self.layers[-1](x)
        
        return x
    
    def evaluate_actions(self,
                        app_embedding: torch.Tensor,
                        action_schemes: List[List[int]],
                        model_path: str = "wordcount_real_rl_model.pt") -> torch.Tensor:
        """
        为同一应用评估多个并行度方案。
        初次调用时加载训练好的模型，后续调用复用已加载的模型。
        
        Args:
            app_embedding: 应用嵌入 [app_embedding_dim]
            action_schemes: 并行度方案列表
            model_path: 训练好的模型文件路径
            
        Returns:
            torch.Tensor: 每个方案的动作价值 [num_schemes]
        """
        if not action_schemes:
            return torch.tensor([])
        
        # 初次加载模型或模型路径发生变化时重新加载
        if not self._model_loaded or self._loaded_model_path != model_path:
            self._load_model_once(model_path)
        
        # 转换为张量
        parallelism_tensor = torch.tensor(action_schemes, dtype=torch.float32)
        
        # 扩展应用嵌入以匹配批量大小
        batch_size = len(action_schemes)
        app_embedding_batch = app_embedding.unsqueeze(0).expand(batch_size, -1)
        
        # 设置为评估模式并进行预测
        self.eval()
        with torch.no_grad():
            values = self.forward(app_embedding_batch, parallelism_tensor)
        
        return values.squeeze(1)
    
    def _load_model_once(self, model_path: str):
        """
        一次性加载模型，避免重复加载。
        
        Args:
            model_path: 模型文件路径
        """
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"⚠️  警告: 训练好的模型文件不存在: {model_path}")
            print("   使用随机初始化的模型进行评估，结果可能不准确")
            print("   建议先运行训练生成模型文件")
            self._model_loaded = True  # 标记为已尝试加载，避免重复警告
            self._loaded_model_path = model_path
            return
        
        # 加载训练好的模型参数
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            if 'action_evaluator_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['action_evaluator_state_dict'])
                print(f"✓ 成功加载训练好的模型: {model_path}")
                self._model_loaded = True
                self._loaded_model_path = model_path
            else:
                print(f"❌ 模型文件格式错误，未找到action_evaluator_state_dict: {model_path}")
                print("   将使用随机初始化的模型")
                self._model_loaded = True  # 标记为已尝试加载
                self._loaded_model_path = model_path
        except Exception as e:
            print(f"❌ 加载模型失败: {str(e)}")
            print("   将使用随机初始化的模型")
            self._model_loaded = True  # 标记为已尝试加载
            self._loaded_model_path = model_path
    
    def select_best_action(self,
                          app_embedding: torch.Tensor,
                          action_schemes: List[List[int]],
                          model_path: str = "wordcount_real_rl_model.pt") -> Tuple[List[int], float]:
        """
        从候选方案中选择最佳并行度方案。
        
        Args:
            app_embedding: 应用嵌入
            action_schemes: 候选并行度方案列表
            model_path: 训练好的模型文件路径
            
        Returns:
            Tuple[List[int], float]: 最佳方案及其价值
        """
        if not action_schemes:
            raise ValueError("未提供动作方案")
        
        
        
        values = self.evaluate_actions(app_embedding, action_schemes, model_path)

        rand_prod = random.random()

        if rand_prod < 0.1:
            # 30%概率随机选择一个方案，增加探索
            best_idx = random.randint(0, len(action_schemes) - 1)
        else:
            best_idx = torch.argmax(values).item()
            
        best_scheme = action_schemes[best_idx]
        best_value = values[best_idx].item()
        
        return best_scheme, best_value
    
    def is_model_loaded(self) -> bool:
        """
        检查模型是否已加载。
        
        Returns:
            bool: 是否已加载模型
        """
        return self._model_loaded
    
    def get_loaded_model_path(self) -> Optional[str]:
        """
        获取已加载的模型路径。
        
        Returns:
            Optional[str]: 已加载的模型路径，如果未加载则返回None
        """
        return self._loaded_model_path if self._model_loaded else None
    
    def reset_model_state(self):
        """
        重置模型加载状态，强制下次调用时重新加载模型。
        """
        self._model_loaded = False
        self._loaded_model_path = None
    
    def load_pretrained_model(self, model_path: str) -> bool:
        """
        加载预训练模型（兼容旧接口）。
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否成功加载
        """
        # 重置状态并强制加载
        self.reset_model_state()
        self._load_model_once(model_path)
        return self._model_loaded and os.path.exists(model_path)
    
    def try_load_default_model(self, default_path: str = "wordcount_real_rl_model.pt") -> bool:
        """
        尝试加载默认模型。
        
        Args:
            default_path: 默认模型路径
            
        Returns:
            bool: 是否成功加载
        """
        if self._model_loaded:
            return os.path.exists(self._loaded_model_path or "")  # 已经加载过模型
            
        return self.load_pretrained_model(default_path)
    
    def is_model_available(self) -> bool:
        """
        检查模型是否可用。
        
        Returns:
            bool: 模型是否已加载并可用
        """
        return self._model_loaded and os.path.exists(self._loaded_model_path or "")


class ActionEvaluatorTrainer:
    """
    动作评估网络的训练器。
    """
    
    def __init__(self,
                 evaluator: ActionEvaluator,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        """
        初始化训练器。
        
        Args:
            evaluator: 要训练的ActionEvaluator网络
            learning_rate: 优化器的学习率
            weight_decay: 正则化的权重衰减
        """
        self.evaluator = evaluator
        self.optimizer = torch.optim.Adam(
            evaluator.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_step(self,
                   app_embeddings: torch.Tensor,
                   parallelism_configs: torch.Tensor,
                   target_values: torch.Tensor) -> float:
        """
        执行单次训练步骤。
        
        Args:
            app_embeddings: 应用嵌入批次
            parallelism_configs: 并行度配置批次  
            target_values: 配置的目标值
            
        Returns:
            float: 训练损失
        """
        self.evaluator.train()
        self.optimizer.zero_grad()
        
        # 前向传播
        predicted_values = self.evaluator(app_embeddings, parallelism_configs)
        
        # 计算损失
        loss = self.criterion(predicted_values.squeeze(), target_values)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.evaluator.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self,
                app_embeddings: torch.Tensor,
                parallelism_configs: torch.Tensor,
                target_values: torch.Tensor) -> float:
        """
        验证步骤。
        
        Args:
            app_embeddings: 验证用应用嵌入
            parallelism_configs: 验证用并行度配置
            target_values: 验证用目标值
            
        Returns:
            float: 验证损失
        """
        self.evaluator.eval()
        
        with torch.no_grad():
            predicted_values = self.evaluator(app_embeddings, parallelism_configs)
            loss = self.criterion(predicted_values.squeeze(), target_values)
        
        return loss.item()
    
    def train_epoch(self, 
                   train_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                   val_data: Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None) -> Tuple[float, Optional[float]]:
        """
        训练一个epoch。
        
        Args:
            train_data: (app_embedding, parallelism_config, target_value) 元组列表
            val_data: 相同格式的可选验证数据
            
        Returns:
            Tuple[float, Optional[float]]: 训练损失和验证损失
        """
        total_train_loss = 0.0
        num_batches = 0
        
        for app_emb, para_config, target in train_data:
            loss = self.train_step(app_emb, para_config, target)
            total_train_loss += loss
            num_batches += 1
        
        avg_train_loss = total_train_loss / max(num_batches, 1)
        
        val_loss = None
        if val_data:
            total_val_loss = 0.0
            val_batches = 0
            
            for app_emb, para_config, target in val_data:
                loss = self.validate(app_emb, para_config, target)
                total_val_loss += loss
                val_batches += 1
            
            val_loss = total_val_loss / max(val_batches, 1)
            self.scheduler.step(val_loss)
        
        return avg_train_loss, val_loss