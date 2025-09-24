# LLM-Assisted Reinforcement Learning for Parallelism Scaling in   Stream Computing Systems

## Overview
This repository presents La-Stream, a framework that configures operator parallelism at runtime using a learned exploration-free model, enabling reliable scaling decisions without iterative trial-and-error. Leveraging LLM guidance, La-Stream generates high-quality candidate actions, effectively constraining the search space and avoiding costly online exploration.

DRRS is implemented as a research prototype on top of Apache Storm.

# Core Components

1. **GCN Memory Network** (`gcn_memory.py`)
- Extracts embedding features for streaming applications based on a graph convolutional network (GCN)
- Handles dependencies and adjacency matrices between operators
- Compresses graph structures into one-dimensional embedding vectors

2. **DeepSeek Action Generator** (`deepseek_action_generator.py`)
- Integrates the DeepSeek API to generate parallelism configurations
- Considers operator characteristics such as processing power, input rate, and resource load
- Generates multiple candidate parallelism configurations (10 by default)

3. **Action Evaluator** (`action_evaluator.py`)
- Evaluates action quality using a feedforward neural network
- Combines GCN embedding features with parallelism configurations
- Predicts performance scores for each configuration

4. **RL Agent** (`rl_agent.py`)
- Main controller that integrates all components
- Implements experience replay and model training
- Provides a complete reinforcement learning pipeline




## Prerequisites
You need to design a streaming application that can run on Apache Storm. For details, please refer to the link: <a id="custom-anchor">[[这是我想要跳转到的内容](https://github.com/apache/storm/tree/master/examples/storm-starter)](https://github.com/apache/storm/tree/master/examples/storm-starter)</a> 

## Installation

```bash
pip install -r requirements.txt
```



### Basic Usage

```python
from rl_agent import StreamProcessingRLAgent

# Initialize the agent
agent = StreamProcessingRLAgent(
    deepseek_api_key="your-deepseek-api-key",
    gcn_config={'output_dim': 64},
    deepseek_config={'max_parallelism': 8}
)

# Define stream application operator information
operators_info = [
    {
        'name': 'Source',
        'processing_capability': 0.9,
        'input_rate': 0.8,
        'resource_load': 0.3
    },
    # ... 
]

# Defining dependencies
dependencies = [(0, 1), (1, 2)]  # operator 0 → operator 1 → operator 2

# Choosing the best parallelism scheme
best_scheme, action_info = agent.act(
    operators_info=operators_info,
    dependencies=dependencies
)

print(f"Recommended parallelism configuration: {best_scheme}")
```


