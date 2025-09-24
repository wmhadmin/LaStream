<<<<<<< HEAD
# LaStream
=======
# 分布式流计算RL智能体 (Stream Processing RL Agent)

基于GCN和DeepSeek的强化学习智能体，用于优化分布式流计算系统的算子并行度配置。

## 项目概述

该项目实现了一个智能体，能够自动优化流计算应用程序（如WordCount）中算子的并行度配置，以提升系统性能。

### 核心组件

1. **GCN记忆网络** (`gcn_memory.py`)
   - 基于图卷积网络（GCN）提取流应用程序的嵌入特征
   - 处理算子间的依赖关系和邻接矩阵
   - 将图结构压缩为一维嵌入向量

2. **DeepSeek动作生成器** (`deepseek_action_generator.py`)
   - 集成DeepSeek API生成并行度配置方案
   - 考虑算子的处理能力、输入速率、资源负载等特性
   - 生成多种候选并行度方案（默认10种）

3. **动作评估器** (`action_evaluator.py`)
   - 前馈神经网络评估动作质量
   - 结合GCN嵌入特征和并行度配置
   - 预测每个方案的性能得分

4. **RL智能体** (`rl_agent.py`)
   - 整合所有组件的主控制器
   - 实现经验回放和模型训练
   - 提供完整的强化学习流程

## 系统架构

```
流应用状态 → GCN网络 → 应用嵌入
                ↓
DeepSeek API → 候选并行度方案
                ↓
动作评估器 → 最佳方案选择 → 执行动作
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 主要依赖
- PyTorch >= 1.9.0
- PyTorch Geometric >= 2.0.0  
- OpenAI >= 1.3.0 (用于DeepSeek API)
- NumPy >= 1.21.0

## 使用方法

### 1. 基础使用

```python
from rl_agent import StreamProcessingRLAgent

# 初始化智能体
agent = StreamProcessingRLAgent(
    deepseek_api_key="your-deepseek-api-key",
    gcn_config={'output_dim': 64},
    deepseek_config={'max_parallelism': 8}
)

# 定义流应用算子信息
operators_info = [
    {
        'name': 'Source',
        'processing_capability': 0.9,
        'input_rate': 0.8,
        'resource_load': 0.3
    },
    # ... 更多算子
]

# 定义依赖关系
dependencies = [(0, 1), (1, 2)]  # 算子0 → 算子1 → 算子2

# 选择最佳并行度方案
best_scheme, action_info = agent.act(
    operators_info=operators_info,
    dependencies=dependencies
)

print(f"推荐并行度配置: {best_scheme}")
```

### 2. WordCount示例

运行WordCount流应用优化示例：

```bash
# 设置DeepSeek API密钥
export DEEPSEEK_API_KEY="your-api-key-here"

# 运行示例
python wordcount_example.py
```

### 3. 运行测试

```bash
python test_agent.py
```

## API配置

### DeepSeek API设置

1. 注册DeepSeek账号：https://api.deepseek.com
2. 获取API密钥
3. 设置环境变量：`export DEEPSEEK_API_KEY="your-key"`

### 配置参数

```python
# GCN网络配置
gcn_config = {
    'input_dim': 64,           # 输入特征维度
    'hidden_dims': [128, 256], # 隐藏层维度
    'output_dim': 64,          # 输出嵌入维度
    'dropout': 0.1             # Dropout率
}

# 动作评估器配置
action_evaluator_config = {
    'hidden_dims': [256, 512, 256, 128],  # 隐藏层
    'dropout': 0.2,                       # Dropout率  
    'learning_rate': 0.001                # 学习率
}

# DeepSeek配置
deepseek_config = {
    'max_parallelism': 16,     # 最大并行度
    'model_name': 'deepseek-chat'  # 模型名称
}
```

## 核心特性

### 1. 图神经网络记忆
- 使用GCN提取流应用的拓扑特征
- 支持任意复杂的算子依赖关系
- 自动处理邻接矩阵和节点特征

### 2. 智能动作生成
- 基于算子特性生成合理的并行度方案
- 考虑处理能力、负载均衡、资源约束
- 支持多样化方案生成

### 3. 性能评估与选择
- 深度神经网络预测方案性能
- 自动选择最优配置
- 支持在线学习和模型更新

### 4. 经验回放训练
- 存储历史经验用于模型训练
- 支持批量训练和验证
- 自动调整学习策略

## 文件结构

```
LmModel/
├── requirements.txt          # 依赖包列表
├── gcn_memory.py            # GCN记忆网络
├── deepseek_action_generator.py  # DeepSeek动作生成
├── action_evaluator.py      # 动作评估器
├── rl_agent.py              # 主智能体
├── wordcount_example.py     # WordCount示例
├── test_agent.py           # 单元测试
└── README.md               # 项目文档
```

## 示例结果

WordCount应用优化示例输出：
```
=== Episode 1/5 ===
Selected parallelism scheme: [2, 4, 3, 1]
Predicted value: 0.7543
Reward: 15.2341
System throughput: 89.32

=== Final Results ===
Average reward: 16.8451
Best reward: 18.9234
Reward improvement: +3.6893
```

## 扩展说明

### 添加新的流应用

1. 定义算子信息和依赖关系
2. 创建对应的图数据结构
3. 调用智能体进行优化

### 自定义奖励函数

在环境类中重写`_simulate_performance`方法：

```python
def _simulate_performance(self, parallelism_scheme):
    # 自定义性能评估逻辑
    # 考虑吞吐量、延迟、资源使用等指标
    reward = custom_reward_function(parallelism_scheme)
    return reward, metrics
```




>>>>>>> 5905384 (LaStream Model)
