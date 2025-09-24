import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
import json
import time
from collections import deque
import random

from gcn_memory import GCNMemory, StreamingApplicationEncoder
from deepseek_action_generator import DeepSeekActionGenerator
from action_evaluator import ActionEvaluator, ActionEvaluatorTrainer


class StreamProcessingRLAgent:
    """
    流处理应用并行度优化的强化学习智能体。
    
    该智能体使用：
    1. 基于GCN的内存来提取应用嵌入
    2. DeepSeek生成并行度动作方案
    3. 前馈神经网络评估并选择最佳动作
    """
    
    def __init__(self,
                 deepseek_api_key: str,
                 gcn_config: Dict = None,
                 action_evaluator_config: Dict = None,
                 deepseek_config: Dict = None,
                 experience_buffer_size: int = 3000,
                 device: str = "cpu",
                 pretrained_model_path: str = None):
        """
        初始化强化学习智能体。
        
        Args:
            deepseek_api_key: DeepSeek服务的API密钥
            gcn_config: GCN内存网络的配置
            action_evaluator_config: 动作评估器的配置
            deepseek_config: DeepSeek动作生成器的配置
            experience_buffer_size: 经验回放缓冲区大小
            device: 模型运行设备 (cpu/cuda)
            pretrained_model_path: 预训练模型路径
        """
        self.device = torch.device(device)
        self.logger = logging.getLogger(__name__)
        
        # 如果未提供配置，则使用默认配置初始化组件
        gcn_config = gcn_config or {}
        action_evaluator_config = action_evaluator_config or {}
        deepseek_config = deepseek_config or {}
        
        # 初始化GCN内存
        self.gcn_memory = GCNMemory(
            input_dim=gcn_config.get('input_dim', 64),
            hidden_dims=gcn_config.get('hidden_dims', [128, 256, 128]),
            output_dim=gcn_config.get('output_dim', 64),
            dropout=gcn_config.get('dropout', 0.1)
        ).to(self.device)
        
        # 初始化应用编码器
        self.app_encoder = StreamingApplicationEncoder(self.gcn_memory)
        
        # 初始化DeepSeek动作生成器
        self.action_generator = DeepSeekActionGenerator(
            api_key=deepseek_api_key,
            model_name=deepseek_config.get('model_name', 'deepseek-chat'),
            base_url=deepseek_config.get('base_url', 'https://api.deepseek.com'),
            max_parallelism=deepseek_config.get('max_parallelism', 10)
        )
        
        # 初始化动作评估器
        self.action_evaluator = ActionEvaluator(
            app_embedding_dim=gcn_config.get('output_dim', 64),
            max_parallelism=deepseek_config.get('max_parallelism', 16),
            hidden_dims=action_evaluator_config.get('hidden_dims', [256, 512, 256, 128]),
            dropout=action_evaluator_config.get('dropout', 0.2),
            max_operators=action_evaluator_config.get('max_operators', 10)
        ).to(self.device)
        
        # 初始化动作评估器的训练器
        self.evaluator_trainer = ActionEvaluatorTrainer(
            self.action_evaluator,
            learning_rate=action_evaluator_config.get('learning_rate', 0.001),
            weight_decay=action_evaluator_config.get('weight_decay', 1e-5)
        )
        
        # 用于训练的经验回放缓冲区
        self.experience_buffer = deque(maxlen=experience_buffer_size)
        
        # 系统状态监控缓冲区 (保持150条数据)
        self.system_state_buffer = deque(maxlen=180)
        
        # 训练指标
        self.training_metrics = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'training_losses': []
        }
        
        # 尝试加载预训练模型
        if pretrained_model_path:
            self.load_model(pretrained_model_path)
        else:
            # 尝试加载默认路径的模型
            if self.action_evaluator.try_load_default_model():
                self.logger.info("自动加载了默认预训练模型")
            else:
                self.logger.warning("未找到预训练模型，将使用随机初始化的模型")
        
    def get_application_embedding(self, 
                                operators_info: List[Dict],
                                dependencies: List[Tuple[int, int]]) -> torch.Tensor:
        """
        使用GCN内存提取应用嵌入。
        
        Args:
            operators_info: 操作符信息字典列表
            dependencies: (source_id, target_id) 依赖关系元组列表
            
        Returns:
            torch.Tensor: 应用嵌入向量
        """
        embedding = self.app_encoder.encode_application(operators_info, dependencies)
        return embedding.to(self.device)
    
    def generate_action_schemes(self,
                              operators_info: List[Dict],
                              num_schemes: int = 10) -> List[List[int]]:
        """
        Generate parallelism action schemes using DeepSeek.
        
        Args:
            operators_info: List of operator information
            num_schemes: Number of schemes to generate
            
        Returns:
            List[List[int]]: List of parallelism schemes
        """
        return self.action_generator.generate_parallelism_actions(
            operators_info, num_schemes
        )
    
    def select_best_action(self,
                          app_embedding: torch.Tensor,
                          action_schemes: List[List[int]]) -> Tuple[List[int], float]:
        """
        使用评估网络选择最佳动作方案。
        
        Args:
            app_embedding: GCN的应用嵌入
            action_schemes: 候选并行度方案
            
        Returns:
            Tuple[List[int], float]: 最佳方案及其预测价值
        """
        # 传递模型路径给ActionEvaluator，让它加载训练好的模型
        return self.action_evaluator.select_best_action(app_embedding, action_schemes, "wordcount_real_rl_model.pt")
    
    def act(self,
            operators_info: List[Dict],
            dependencies: List[Tuple[int, int]],
            num_candidate_schemes: int = 10) -> Tuple[List[int], Dict[str, Any]]:
        """
        Main action selection method for the RL agent.
        
        Args:
            operators_info: Current operator information
            dependencies: Operator dependencies
            num_candidate_schemes: Number of candidate schemes to consider
            
        Returns:
            Tuple[List[int], Dict]: Selected parallelism scheme and additional info
        """
        # Step 1: Extract application embedding using GCN
        app_embedding = self.get_application_embedding(operators_info, dependencies)
        
        # Step 2: Generate candidate parallelism schemes using DeepSeek
        candidate_schemes = self.generate_action_schemes(
            operators_info, num_candidate_schemes
        )
        
        # Step 3: Select best action using evaluator network
        best_scheme, predicted_value = self.select_best_action(
            app_embedding, candidate_schemes
        )
        
        # Additional information for analysis
        action_info = {
            'app_embedding': app_embedding.cpu().detach().numpy(),
            'candidate_schemes': candidate_schemes,
            'predicted_value': predicted_value,
            'num_operators': len(operators_info),
            'total_parallelism': sum(best_scheme)
        }
        
        self.logger.info(f"Selected parallelism scheme: {best_scheme} with value: {predicted_value:.4f}")
        
        return best_scheme, action_info
    
    def store_experience(self,
                        operators_info: List[Dict],
                        dependencies: List[Tuple[int, int]],
                        action: List[int],
                        reward: float,
                        next_operators_info: Optional[List[Dict]] = None,
                        next_dependencies: Optional[List[Tuple[int, int]]] = None):
        """
        Store experience in replay buffer for training.
        
        Args:
            operators_info: Current state operator information
            dependencies: Current state dependencies
            action: Selected parallelism scheme
            reward: Observed reward
            next_operators_info: Next state operator information (optional)
            next_dependencies: Next state dependencies (optional)
        """
        experience = {
            'state': {
                'operators_info': operators_info.copy(),
                'dependencies': dependencies.copy()
            },
            'action': action.copy(),
            'reward': reward,
            'next_state': {
                'operators_info': next_operators_info.copy() if next_operators_info else None,
                'dependencies': next_dependencies.copy() if next_dependencies else None
            }
        }
        
        self.experience_buffer.append(experience)

    def store_system_state(self,
                          operators_info: List[Dict],
                          dependencies: List[Tuple[int, int]],
                          action: List[int],
                          reward: float):
        """
        存储系统状态到system_state_buffer
        这个方法与environment的system_state_buffer分开,专门用于agent的训练数据
        """
        system_state = {
            'timestamp': time.time(),
            'operators_info': operators_info.copy(),
            'dependencies': dependencies.copy(),
            'action': action.copy(),
            'reward': reward
        }

        self.system_state_buffer.append(system_state)

    def train_evaluator(self,
                       batch_size: int = 32,
                       num_epochs: int = 1,
                       system_state_buffer: Optional[List[Dict]] = None) -> Dict[str, float]:
        """
        Train the action evaluator using experience replay.
        
        Args:
            batch_size: Training batch size
            num_epochs: Number of training epochs
            
        Returns:
            Dict[str, float]: Training metrics
        """
        # 合并所有可用的训练数据
        all_experiences = []

        # 添加experience_buffer的数据
        all_experiences.extend(list(self.experience_buffer))

        # 添加agent自身的system_state_buffer数据
        # all_experiences.extend(list(self.system_state_buffer))

        # 添加外部传入的system_state_buffer数据（如果有的话）
        if system_state_buffer:
            all_experiences.extend(system_state_buffer)

        if len(all_experiences) < batch_size:
            self.logger.warning("Not enough experiences for training")
            return {'train_loss': 0.0, 'val_loss': 0.0}
        
        # Sample experiences
        self.logger.info(f"Training with {len(all_experiences)} total experiences "
                        f"(experience_buffer: {len(self.experience_buffer)}, "
                        f"system_state_buffer: {len(self.system_state_buffer)}, "
                        f"external_buffer: {len(system_state_buffer) if system_state_buffer else 0})")

        # Sample experiences from all available data
        sample_size = min(batch_size * 5, len(all_experiences))
        experiences = random.sample(all_experiences, sample_size)
        
        # Prepare training data
        train_data = []
        
        for exp in experiences:
            try:
                # 处理不同格式的经验数据
                if 'state' in exp:
                    # experience_buffer格式的数据
                    operators_info = exp['state']['operators_info']
                    dependencies = exp['state']['dependencies']
                    action = exp['action']
                    reward = exp['reward']
                else:
                    # system_state_buffer格式的数据
                    operators_info = exp['operators_info']
                    dependencies = exp['dependencies']
                    action = exp.get('action', exp.get('current_parallelism', []))
                    reward = exp['reward']

                # Get application embedding for current state
                app_embedding = self.get_application_embedding(
                    operators_info,
                    dependencies
                )

                # Convert action to tensor
                action_tensor = torch.tensor(action, dtype=torch.float32)

                # Use reward as target value
                target_value = torch.tensor(reward, dtype=torch.float32)

                train_data.append((
                    app_embedding.unsqueeze(0),
                    action_tensor.unsqueeze(0),
                    target_value.unsqueeze(0)
                ))

            except Exception as e:
                self.logger.warning(f"Skipping invalid experience data: {e}")
        
        # Split into train/validation
        split_idx = int(0.8 * len(train_data))
        train_split = train_data[:split_idx]
        val_split = train_data[split_idx:] if split_idx < len(train_data) else []
        
        # Training loop
        total_train_loss = 0.0
        total_val_loss = 0.0
        
        for epoch in range(num_epochs):
            train_loss, val_loss = self.evaluator_trainer.train_epoch(
                train_split, val_split if val_split else None
            )
            total_train_loss += train_loss
            if val_loss is not None:
                total_val_loss += val_loss
        
        avg_train_loss = total_train_loss / num_epochs
        avg_val_loss = total_val_loss / num_epochs if val_split else 0.0
        
        # Update training metrics
        self.training_metrics['training_losses'].append(avg_train_loss)
        
        self.logger.info(f"Training completed - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'num_experiences': len(self.experience_buffer)
        }
    
    def update_metrics(self, episode_reward: float):
        """Update training metrics."""
        self.training_metrics['episodes'] += 1
        self.training_metrics['total_reward'] += episode_reward
        self.training_metrics['avg_reward'] = (
            self.training_metrics['total_reward'] / self.training_metrics['episodes']
        )
    
    def save_model(self, filepath: str):
        """Save model components to file."""
        checkpoint = {
            'gcn_memory_state_dict': self.gcn_memory.state_dict(),
            'action_evaluator_state_dict': self.action_evaluator.state_dict(),
            'optimizer_state_dict': self.evaluator_trainer.optimizer.state_dict(),
            'training_metrics': self.training_metrics
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model components from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.gcn_memory.load_state_dict(checkpoint['gcn_memory_state_dict'])
        self.action_evaluator.load_state_dict(checkpoint['action_evaluator_state_dict'])
        self.evaluator_trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_metrics = checkpoint.get('training_metrics', self.training_metrics)
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def load_pretrained_evaluator(self, model_path: str) -> bool:
        """
        加载预训练的动作评估器。
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            bool: 是否成功加载
        """
        return self.action_evaluator.load_pretrained_model(model_path)
    
    def is_evaluator_trained(self) -> bool:
        """
        检查动作评估器是否已训练。
        
        Returns:
            bool: 是否已训练
        """
        return self.action_evaluator.is_model_available()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics."""
        return {
            'episodes': self.training_metrics['episodes'],
            'average_reward': self.training_metrics['avg_reward'],
            'total_reward': self.training_metrics['total_reward'],
            'experience_buffer_size': len(self.experience_buffer),
            'recent_losses': self.training_metrics['training_losses'][-10:] if self.training_metrics['training_losses'] else []
        }
    
    def reset_training_stats(self):
        """Reset training statistics."""
        self.training_metrics = {
            'episodes': 0,
            'total_reward': 0.0,
            'avg_reward': 0.0,
            'training_losses': []
        }
        self.experience_buffer.clear()
        self.logger.info("Training statistics reset")