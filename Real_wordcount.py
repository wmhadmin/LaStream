#!/usr/bin/env python3
"""
Real WordCount Environment - Connect to real distributed streaming environment

This module reads real-time metrics from Redis for WordCount streaming application
and converts them to RL agent format: operators_info, dependencies, system_latency, inflight_data, total_instances

WordCount application has 3 operators:
- Source operator (sentence-spout): parallelism 2
- Splitter operator (splitter-bolt): parallelism 4  
- Counter operator (counter-bolt): parallelism 4

Redis instance keys:
counter-bolt-5, sentence-spout-8, counter-bolt-4, splitter-bolt-11, splitter-bolt-10,
sentence-spout-9, splitter-bolt-12, counter-bolt-6, splitter-bolt-13, counter-bolt-7
"""

import logging
import time
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import threading

import torch
import math
import os
import subprocess
import time

from rl_agent import StreamProcessingRLAgent
from gcn_memory import create_wordcount_graph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_reward(L: float, D: float, parallelism_scheme: List[int], 
                    target_latency: float = 100.0, 
                    target_data_constraint: float = 1000.0,
                    min_parallelism: int = 1,
                    eta_l: float = 0.4, 
                    eta_d: float = 0.4, 
                    eta_a: float = 0.2) -> float:
    """
    Calculate reward value according to the paper formula.
    
    r = ∛(e^(-η_l·L/θ) × e^(-η_d·D/D*) × e^(-η_a/n·Σ(a_c,i/a_c,0)))
    
    Args:
        L: System latency
        D: In-flight data amount
        parallelism_scheme: Parallelism scheme [a_c,1, a_c,2, ..., a_c,n]
        target_latency: User-defined target latency θ
        target_data_constraint: Data accumulation constraint D*
        min_parallelism: Minimum operator parallelism a_c,0
        eta_l: Latency importance parameter
        eta_d: Data amount importance parameter
        eta_a: Parallelism importance parameter
        
    Returns:
        float: Reward value
    """
    n = len(parallelism_scheme)  # Number of operators
    
    # First term: latency penalty term e^(-η_l·L/θ)
    latency_term = math.exp(-eta_l * L / target_latency)
    
    # Second term: data accumulation penalty term e^(-η_d·D/D*)
    data_term = math.exp(-eta_d * D / target_data_constraint)
    
    # Third term: parallelism cost term e^(-η_a/n·Σ(a_c,i/a_c,0))
    parallelism_sum = sum(parallelism / min_parallelism for parallelism in parallelism_scheme)
    parallelism_term = math.exp(-eta_a / n * parallelism_sum)
    
    # Calculate final reward: geometric mean of three terms (cube root)
    reward = (latency_term * data_term * parallelism_term) ** (1/3)
    
    return reward


# Try to import redis, but make it optional for testing
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis module not available - using mock data for testing")
    REDIS_AVAILABLE = False


@dataclass
class InstanceMetrics:
    """Metrics data for a single operator instance"""
    instance_key: str
    operator_name: str
    component_type: str  # bolt or spout
    task_id: int
    output_count: float
    processing_capacity: float
    cpu_usage: float
    memory_usage: float



@dataclass
class OperatorMetrics:
    """Aggregated metrics at operator level"""
    operator_name: str
    operator_id: int
    component_type: str  # Component type: "bolt" or "spout" 
    output_count: float  # Sum of all instances output_count
    processing_capacity: float  # Average of all instances processing_capacity
    cpu_usage: float  # Sum of all instances cpu_usage
    memory_usage: float  # Sum of all instances memory_usage
    instance_count: int  # Number of instances (parallelism)
    dependencies: List[int] = None  # List of operator IDs this operator depends on
    input_rate: float = 0.0  # Sum of output_count from dependent operators


class WordCountRealEnvironment:
    """
    WordCount real environment that connects to Redis for real-time metrics
    """
    
    def __init__(self, 
                 redis_host: str = "192.168.103.100",
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 redis_password: Optional[str] = None):
        """
        Initialize real environment
        
        Args:
            redis_host: Redis server IP
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Redis password (if required)
        """
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        
        # Initialize Redis connection if available
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                self.redis_client.ping()
                logger.info(f"Successfully connected to Redis: {redis_host}:{redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Instance keys will be dynamically fetched from Redis
        self.instance_keys = []
        
        # Operator mapping: operator_name -> operator_id
        self.operator_mapping = {
            "sentence": 0,  # Source operator
            "splitter": 1,  # Splitter operator
            "counter": 2    # Counter operator
        }
        
        # Dependencies: (source_operator_id, target_operator_id)
        self.dependencies = [(0, 1), (1, 2)]  # Source -> Splitter -> Counter

        # Global operator instance count configuration
        self.operator_instance_counts = {
            "sentence": 2,  # Source operator
            "splitter": 4,  # Splitter operator
            "counter": 4    # Counter operator
        }
        
        # Mock data for testing when Redis is not available
        self.mock_data = {
            "counter-bolt-5": {"output-count": "1200", "processing-capacity": "85.5", "cpu-usage": "45.2", "memory-usage": "67.8"},
            "sentence-spout-8": {"output-count": "800", "processing-capacity": "90.2", "cpu-usage": "35.6", "memory-usage": "52.3", 
                                "completeLatency": "95.5", "spout-pending": "120.0"},
            "counter-bolt-4": {"output-count": "1180", "processing-capacity": "83.1", "cpu-usage": "48.7", "memory-usage": "69.2"},
            "splitter-bolt-11": {"output-count": "4200", "processing-capacity": "78.9", "cpu-usage": "72.3", "memory-usage": "58.9"},
            "splitter-bolt-10": {"output-count": "4150", "processing-capacity": "76.4", "cpu-usage": "75.6", "memory-usage": "61.2"},
            "sentence-spout-9": {"output-count": "850", "processing-capacity": "88.7", "cpu-usage": "33.2", "memory-usage": "48.9",
                                "completeLatency": "102.3", "spout-pending": "135.0"},
            "splitter-bolt-12": {"output-count": "4300", "processing-capacity": "79.8", "cpu-usage": "68.7", "memory-usage": "59.8"},
            "counter-bolt-6": {"output-count": "1250", "processing-capacity": "84.6", "cpu-usage": "46.8", "memory-usage": "71.4"},
            "splitter-bolt-13": {"output-count": "4180", "processing-capacity": "77.2", "cpu-usage": "74.1", "memory-usage": "62.7"},
            "counter-bolt-7": {"output-count": "1190", "processing-capacity": "82.8", "cpu-usage": "47.5", "memory-usage": "68.6"}
        }
        
        # Storm configuration
        self.nimbus_host = "192.168.103.100"  # Remote Nimbus host
        self.nimbus_port = 6627  # Default Nimbus port
        self.topology_name = "word-count-topology"  # Default topology name
        self.remote_user = "root"  # SSH user for remote execution
        self.storm_bin_path = ""  # Storm binary path on remote host

        # System state monitoring
        self.system_state_buffer = deque(maxlen=150)
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_lock = threading.Lock()
        
        # Reward calculation parameters
        self.reward_params = {
            'target_latency': 100.0,
            'target_data_constraint': 1000.0,
            'min_parallelism': 1,
            'eta_l': 1.0,
            'eta_d': 1.0,
            'eta_a': 0.5
        }
        
        logger.info("WordCount real environment initialized")

        # Start system state monitoring
        self.start_system_monitoring()
    
    def configure_remote_storm(self, 
                              nimbus_host: str = "192.168.103.100",
                              nimbus_port: int = 6627,
                              topology_name: str = "word-count-topology",
                              remote_user: str = "storm",
                              storm_bin_path: str = "/opt/storm/bin/storm"):
        """
        Configure remote Storm cluster connection parameters
        
        Args:
            nimbus_host: IP address of the Nimbus host
            nimbus_port: Nimbus port (default 6627)
            topology_name: Name of the topology to rebalance
            remote_user: SSH username for remote execution
            storm_bin_path: Path to storm binary on remote host
        """
        self.nimbus_host = nimbus_host
        self.nimbus_port = nimbus_port
        self.topology_name = topology_name
        self.remote_user = remote_user
        self.storm_bin_path = storm_bin_path
        
        logger.info(f"Storm configuration updated:")
        logger.info(f"  Nimbus host: {self.nimbus_host}:{self.nimbus_port}")
        logger.info(f"  Topology: {self.topology_name}")
        logger.info(f"  Remote user: {self.remote_user}")
        logger.info(f"  Storm binary: {self.storm_bin_path}")
    
    def _get_all_instance_keys(self) -> List[str]:
        """Dynamically get all instance keys from Redis"""
        if self.redis_client:
            try:
                # Get all keys matching bolt and spout patterns
                bolt_keys = self.redis_client.keys('*-bolt-*')
                spout_keys = self.redis_client.keys('*-spout-*')
                all_keys = bolt_keys + spout_keys
                # logger.info(f"Dynamically found {len(all_keys)} instance keys from Redis")
                return all_keys
            except Exception as e:
                logger.warning(f"Failed to get keys from Redis: {e}")
                return list(self.mock_data.keys())
        else:
            return list(self.mock_data.keys())
    
    def _get_instance_data(self, key: str) -> Dict[str, str]:
        """Get data for a specific instance, from Redis or mock data"""
        if self.redis_client:
            try:
                return self.redis_client.hgetall(key)
            except Exception as e:
                logger.warning(f"Failed to get Redis data for {key}: {e}")
                return self.mock_data.get(key, {})
        else:
            return self.mock_data.get(key, {})
    
    def _fetch_instance_metrics(self) -> List[InstanceMetrics]:
        """
        Fetch metrics for all operator instances from Redis
        
        Returns:
            List[InstanceMetrics]: List of all instance metrics
        """
        instance_metrics = []
        
        # logger.info("Starting to fetch instance metrics from Redis...")
        
        # Dynamically get all instance keys from Redis
        self.instance_keys = self._get_all_instance_keys()
        
        for key in self.instance_keys:
            try:
                # Parse key format: "operator-component-instanceID"
                # Example: counter-bolt-5 -> operator: counter, component: bolt, instanceID: 5
                parts = key.split('-')
                if len(parts) >= 3:
                    operator_name = parts[0]  # e.g., "counter"
                    component_type = parts[1]  # e.g., "bolt" or "spout"
                    task_id = int(parts[2])  # e.g., 5
                    
                    # Get instance data
                    data = self._get_instance_data(key)
                    
                    if data:
                        output_count = float(data.get("output-count", 0))
                        processing_capacity = float(data.get("processing-capacity", 0))
                        cpu_usage = float(data.get("cpu-usage", 0))
                        memory_usage = float(data.get("memory-usage", 0))
                        
                        instance_metric = InstanceMetrics(
                            instance_key=key,
                            operator_name=operator_name,
                            component_type=component_type,
                            task_id=task_id,
                            output_count=output_count,
                            processing_capacity=processing_capacity,
                            cpu_usage=cpu_usage,
                            memory_usage=memory_usage
                        )
                        
                        instance_metrics.append(instance_metric)
                        logger.debug(f"Got instance {key}: output={output_count}, capacity={processing_capacity}, "
                                   f"cpu={cpu_usage}, memory={memory_usage}")
                    else:
                        logger.warning(f"No data found for instance {key}")
                        
            except Exception as e:
                logger.warning(f"Error processing instance {key}: {e}")
                continue
        
        # logger.info(f"Successfully fetched metrics for {len(instance_metrics)} instances")
        return instance_metrics
    
    def _calculate_operator_metrics(self, instance_metrics: List[InstanceMetrics]) -> List[OperatorMetrics]:
        """
        Calculate aggregated metrics for each operator based on instance metrics
        
        As per your requirements:
        - operator output_count: sum of all instances output_count
        - operator processing_capacity: average of all instances processing_capacity
        - operator cpu_usage: sum of all instances cpu_usage  
        - operator memory_usage: sum of all instances memory_usage
        
        Args:
            instance_metrics: List of instance metrics
            
        Returns:
            List[OperatorMetrics]: List of aggregated operator metrics
        """
        # Group instances by operator name
        operator_instances = defaultdict(list)
        for instance in instance_metrics:
            operator_instances[instance.operator_name].append(instance)
        
        operator_metrics = []
        
        for operator_name, instances in operator_instances.items():
            if operator_name not in self.operator_mapping:
                logger.warning(f"Unknown operator: {operator_name}")
                continue
            
            operator_id = self.operator_mapping[operator_name]
            
            # Calculate aggregated metrics according to requirements
            # Use global instance count configuration instead of calculating from actual instances
            instance_count = self.operator_instance_counts.get(operator_name)
            total_output_count = sum(inst.output_count for inst in instances)
            avg_processing_capacity = sum(inst.processing_capacity for inst in instances) / instance_count if instance_count > 0 else 0.0
            total_cpu_usage = sum(inst.cpu_usage for inst in instances)
            total_memory_usage = sum(inst.memory_usage for inst in instances)
            
            
            # Get component_type from the first instance (all instances of same operator have same component_type)
            component_type = instances[0].component_type
            
            operator_metric = OperatorMetrics(
                operator_name=operator_name,
                operator_id=operator_id,
                component_type=component_type,
                output_count=total_output_count,
                processing_capacity=avg_processing_capacity,
                cpu_usage=total_cpu_usage,
                memory_usage=total_memory_usage,
                instance_count=instance_count
            )
            
            operator_metrics.append(operator_metric)
            
            # logger.info(f"Operator {operator_name} (ID={operator_id}, type={component_type}) aggregated metrics:")
            # logger.info(f"  Instance count: {instance_count}")
            # logger.info(f"  Total output: {total_output_count:.2f}")
            # logger.info(f"  Average processing capacity: {avg_processing_capacity:.2f}")
            # logger.info(f"  Total CPU usage: {total_cpu_usage:.2f}")
            # logger.info(f"  Total memory usage: {total_memory_usage:.2f}")
        
        # Sort by operator_id
        operator_metrics.sort(key=lambda x: x.operator_id)
        
        # Calculate input_rate for each operator based on dependencies
        operator_metrics = self._calculate_input_rates(operator_metrics)
        
        return operator_metrics
    
    def _calculate_input_rates(self, operator_metrics: List[OperatorMetrics]) -> List[OperatorMetrics]:
        """
        Calculate input_rate for each operator based on its dependencies
        input_rate = sum of output_count from all dependent operators
        
        Args:
            operator_metrics: List of operator metrics
            
        Returns:
            List[OperatorMetrics]: Updated operator metrics with input_rate calculated
        """
        # Create a mapping from operator_id to operator metrics for quick lookup
        operator_map = {op.operator_id: op for op in operator_metrics}
        
        # Set dependencies for each operator
        for op_metric in operator_metrics:
            if op_metric.operator_id == 0:
                op_metric.dependencies = []
            elif op_metric.operator_id == 1:
                op_metric.dependencies = [0]
            elif op_metric.operator_id == 2:
                op_metric.dependencies = [1]
            else:
                op_metric.dependencies = []
        
        # Calculate input_rate for each operator
        for op_metric in operator_metrics:
            if op_metric.dependencies:
                # Sum the output_count of all dependent operators
                input_rate = 0.0
                for dep_id in op_metric.dependencies:
                    if dep_id in operator_map:
                        input_rate += operator_map[dep_id].output_count
                        logger.debug(f"Operator {op_metric.operator_name} (ID={op_metric.operator_id}) "
                                   f"receives input {operator_map[dep_id].output_count} from operator {dep_id}")
                op_metric.input_rate = input_rate
                # logger.info(f"Operator {op_metric.operator_name} (ID={op_metric.operator_id}) "
                #            f"total input_rate: {input_rate}")
            else:
                # Source operators have no input (they generate data)
                op_metric.input_rate = 0.0
                # logger.info(f"Source operator {op_metric.operator_name} (ID={op_metric.operator_id}) "
                #            f"has no input (generates data)")
        
        return operator_metrics
    
    def _transform_to_operators_info(self, operator_metrics: List[OperatorMetrics]) -> List[Dict]:
        """
        Transform operator metrics to operators_info format required by RL agent
        
        Args:
            operator_metrics: List of operator metrics
            
        Returns:
            List[Dict]: operators_info format for RL agent
        """
        operators_info = []
        
        for op_metric in operator_metrics:
            # # Normalize metrics to [0,1] range
            # processing_capability = min(op_metric.processing_capacity / 100.0, 1.0)
            # input_rate = min(op_metric.output_count / 5000.0, 1.0)  # Adjust normalization range
            # resource_load = min((op_metric.cpu_usage + op_metric.memory_usage) / 300.0, 1.0)
            
            # # Set selectivity and display name based on operator type
            # if op_metric.operator_name == "sentence":
            #     selectivity = 1.0  # Source operator generates data
            #     operator_display_name = "Source"
            # elif op_metric.operator_name == "splitter":
            #     selectivity = 5.0  # Splitter operator splits sentences into words
            #     operator_display_name = "Splitter"
            # elif op_metric.operator_name == "counter":
            #     selectivity = 0.1  # Counter operator aggregates counts
            #     operator_display_name = "Counter"
            # else:
            #     selectivity = 1.0
            #     operator_display_name = op_metric.operator_name.capitalize()
            
            # Set dependencies based on operator_id
            if op_metric.operator_id == 0:
                op_metric.dependencies = []
            elif op_metric.operator_id == 1:
                op_metric.dependencies = [0]
            elif op_metric.operator_id == 2:
                op_metric.dependencies = [1]
            else:
                op_metric.dependencies = []

            operators_info.append(op_metric)
            
            
        return operators_info
    
    def _calculate_system_metrics(self, operator_metrics: List[OperatorMetrics]) -> Tuple[float, float]:
        """
        Calculate system-level metrics: system latency and in-flight data
        
        System latency comes from spout instances' completeLatency field (average if multiple spouts)
        In-flight data comes from spout instances' spout-pending field (average if multiple spouts)
        
        Args:
            operator_metrics: List of operator metrics
            
        Returns:
            Tuple[float, float]: (system_latency, inflight_data)
        """
        # Get system latency and inflight data from spout instances
        system_latency, inflight_data = self._get_spout_system_metrics()
        
        # Fallback to calculated values if spout data is not available
        if system_latency is None or inflight_data is None:
            logger.warning("Spout system metrics not available, using calculated values")
            
            if not operator_metrics:
                return 100.0, 500.0
            
            # Fallback calculation
            avg_processing_capacity = np.mean([op.processing_capacity for op in operator_metrics])
            system_latency = max(20.0, min(200.0, 150.0 - avg_processing_capacity))
            
            total_output = sum(op.output_count for op in operator_metrics)
            total_cpu_usage = sum(op.cpu_usage for op in operator_metrics)
            cpu_load_factor = total_cpu_usage / (len(operator_metrics) * 100.0)
            output_factor = total_output / 5000.0
            inflight_data = max(100.0, min(2000.0, 300.0 + output_factor * 400.0 + cpu_load_factor * 300.0))
        
        return system_latency, inflight_data
    
    def _get_spout_system_metrics(self) -> Tuple[float, float]:
        """
        Get system latency and inflight data from spout instances
        
        Returns:
            Tuple[float, float]: (system_latency from completeLatency, inflight_data from spout-pending)
                                Returns (None, None) if no spout data available
        """
        if not self.redis_client:
            logger.warning("No Redis connection for spout metrics")
            return None, None
        
        try:
            # Get all spout keys
            spout_keys = self.redis_client.keys('*-spout-*')
            
            if not spout_keys:
                logger.warning("No spout instances found for system metrics")
                return None, None
            
            complete_latencies = []
            spout_pendings = []
            
            for spout_key in spout_keys:
                try:
                    # Get completeLatency for system latency
                    complete_latency = self.redis_client.hget(spout_key, "completeLatency")
                    if complete_latency:
                        complete_latencies.append(float(complete_latency))
                    
                    # Get spout-pending for inflight data
                    spout_pending = self.redis_client.hget(spout_key, "spout-pending")
                    if spout_pending:
                        spout_pendings.append(float(spout_pending))
                        
                    logger.debug(f"Spout {spout_key}: completeLatency={complete_latency}, spout-pending={spout_pending}")
                    
                except Exception as e:
                    logger.warning(f"Error getting spout metrics from {spout_key}: {e}")
                    continue
            
            # Calculate averages if multiple spout instances
            system_latency = None
            inflight_data = None
            
            if complete_latencies:
                system_latency = np.mean(complete_latencies)
                # logger.info(f"System latency from {len(complete_latencies)} spout instances: {system_latency:.2f}ms")
            
            if spout_pendings:
                inflight_data = np.mean(spout_pendings)
                # logger.info(f"Inflight data from {len(spout_pendings)} spout instances: {inflight_data:.2f}")
            
            return system_latency, inflight_data
            
        except Exception as e:
            logger.error(f"Error getting spout system metrics: {e}")
            return None, None
    
    def get_state(self) -> Tuple[List[Dict], List[Tuple[int, int]], float, float, int]:
        """
        Get current environment state
        
        Returns:
            Tuple: (operators_info, dependencies, system_latency, inflight_data, total_instances)
        """
        try:
            # logger.info("Starting to get environment state...")
            
            # 1. Get all instance metrics
            instance_metrics = self._fetch_instance_metrics()
            
            # 2. Calculate operator aggregated metrics
            operator_metrics = self._calculate_operator_metrics(instance_metrics)
            
            # 3. Transform to operators_info format
            # operators_info = self._transform_to_operators_info(operator_metrics)
            
            # 4. Calculate system-level metrics
            system_latency, inflight_data = self._calculate_system_metrics(operator_metrics)
            
            # 5. Calculate total instances
            total_instances = sum(op.instance_count for op in operator_metrics)
            
            # logger.info(f"State retrieved successfully:")
            # logger.info(f"  Number of operators: {len(operator_metrics)}")
            # logger.info(f"  System latency: {system_latency:.1f}ms")
            # logger.info(f"  In-flight data: {inflight_data:.1f}")
            # logger.info(f"  Total instances: {total_instances}")
            
            return operator_metrics, self.dependencies, system_latency, inflight_data, total_instances
            
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            # Return default state
            return self._get_default_state()
    
    def _get_default_state(self) -> Tuple[List[Dict], List[Tuple[int, int]], float, float, int]:
        """Return default state when Redis is not available"""
        operators_info = [
            {
                'name': 'Source',
                'id': 0,
                'processing_capability': 0.5,
                'input_rate': 0.5,
                'resource_load': 0.5,
                'selectivity': 1.0,
                'dependencies': [],
                'current_parallelism': 2
            },
            {
                'name': 'Splitter',
                'id': 1,
                'processing_capability': 0.5,
                'input_rate': 0.5,
                'resource_load': 0.5,
                'selectivity': 5.0,
                'dependencies': [0],
                'current_parallelism': 4
            },
            {
                'name': 'Counter',
                'id': 2,
                'processing_capability': 0.5,
                'input_rate': 0.5,
                'resource_load': 0.5,
                'selectivity': 0.1,
                'dependencies': [1],
                'current_parallelism': 4
            }
        ]
        
        return operators_info, self.dependencies, 100.0, 500.0, 10
    
    def print_detailed_metrics(self):
        """Print detailed metrics information for debugging"""
        logger.info("=== Detailed Metrics Report ===")
        
        # Get instance metrics
        instance_metrics = self._fetch_instance_metrics()
        
        logger.info(f"Total {len(instance_metrics)} instances:")
        for instance in instance_metrics:
            logger.info(f"Instance {instance.instance_key}:")
            logger.info(f"  Operator: {instance.operator_name}")
            logger.info(f"  Output count: {instance.output_count}")
            logger.info(f"  Processing capacity: {instance.processing_capacity}")
            logger.info(f"  CPU usage: {instance.cpu_usage}")
            logger.info(f"  Memory usage: {instance.memory_usage}")
        
        # Calculate and display operator aggregated metrics
        operator_metrics = self._calculate_operator_metrics(instance_metrics)
        
        logger.info(f"\nOperator aggregated metrics:")
        for op in operator_metrics:
            logger.info(f"Operator {op.operator_name} (ID={op.operator_id}):")
            logger.info(f"  Instance count: {op.instance_count}")
            logger.info(f"  Total output: {op.output_count}")
            logger.info(f"  Average processing capacity: {op.processing_capacity:.2f}")
            logger.info(f"  Total CPU usage: {op.cpu_usage}")
            logger.info(f"  Total memory usage: {op.memory_usage}")
    
    def health_check(self) -> Dict[str, Any]:
        """Health check"""
        health_info = {
            'redis_available': REDIS_AVAILABLE,
            'redis_connected': False,
            'data_available': False,
            'instance_count': 0,
            'operators_found': [],
            'error': None
        }
        
        try:
            if self.redis_client:
                # Test Redis connection
                self.redis_client.ping()
                health_info['redis_connected'] = True
            
            # Check data availability
            instance_metrics = self._fetch_instance_metrics()
            health_info['data_available'] = len(instance_metrics) > 0
            health_info['instance_count'] = len(instance_metrics)
            
            operators_found = set(inst.operator_name for inst in instance_metrics)
            health_info['operators_found'] = list(operators_found)
            
        except Exception as e:
            health_info['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health_info
    
    def step(self, parallelism_scheme: List[int]) -> Tuple[float, Dict[str, Any]]:
        """
        Execute parallelism scheme in the streaming system and return reward
        
        Steps:
        1. Apply parallelism scheme via Storm rebalance
        2. Wait 90 seconds for system to complete rebalance
        3. Get updated system metrics from Redis
        4. Calculate reward based on new metrics
        
        Args:
            parallelism_scheme: List of parallelism values for each operator [source, splitter, counter]
            
        Returns:
            Tuple[float, Dict]: (reward, performance_metrics)
        """
        logger.info(f"Starting step with parallelism scheme: {parallelism_scheme}")
        
        try:
            # Step 1: Apply parallelism scheme via Storm rebalance
            rebalance_success = self._apply_rebalance(parallelism_scheme)

            if rebalance_success:
                # Update global operator instance counts based on parallelism_scheme
                operator_names = ["sentence", "splitter", "counter"]
                for i, operator_name in enumerate(operator_names):
                    if i < len(parallelism_scheme):
                        self.operator_instance_counts[operator_name] = parallelism_scheme[i]
                
                # Step 2: Wait for system to complete rebalance
                # logger.info("Waiting 60 seconds for system to complete rebalance...")
                time.sleep(90)
                logger.info(f"Updated global operator instance counts: {self.operator_instance_counts}")
            else:
                logger.warning("Rebalance failed, using current system state for reward calculation")
            
            # Step 3: Get updated system metrics
            system_latency, inflight_data = self._get_updated_system_metrics()
            
            # Step 4: Calculate reward
            reward = calculate_reward(
                L=system_latency,
                D=inflight_data,
                parallelism_scheme=parallelism_scheme,
                **self.reward_params
            )
            
            # Prepare performance metrics
            performance_metrics = {
                'system_latency': system_latency,
                'inflight_data': inflight_data,
                'parallelism_scheme': parallelism_scheme.copy(),
                'reward': reward,
                'rebalance_success': rebalance_success,
                'total_instances': sum(parallelism_scheme),
                'reward_components': {
                    'latency_term': math.exp(-self.reward_params['eta_l'] * system_latency / self.reward_params['target_latency']),
                    'data_term': math.exp(-self.reward_params['eta_d'] * inflight_data / self.reward_params['target_data_constraint']),
                    'parallelism_term': math.exp(-self.reward_params['eta_a'] / len(parallelism_scheme) * 
                                               sum(p / self.reward_params['min_parallelism'] for p in parallelism_scheme))
                }
            }
            
            logger.info(f"Step completed - Reward: {reward:.6f}, Latency: {system_latency:.2f}ms, "
                       f"Inflight: {inflight_data:.2f}, Total instances: {sum(parallelism_scheme)}")
            
            return reward, performance_metrics
            
        except Exception as e:
            logger.error(f"Error in step execution: {e}")
            # Return default values on error
            return 0.0, {
                'system_latency': 100.0,
                'inflight_data': 500.0,
                'parallelism_scheme': parallelism_scheme.copy(),
                'reward': 0.0,
                'rebalance_success': False,
                'error': str(e)
            }
    
    def _apply_rebalance(self, parallelism_scheme: List[int]) -> bool:
        """
        Apply parallelism scheme using Storm rebalance command
        Supports both local and remote execution via SSH
        
        Args:
            parallelism_scheme: [source_parallelism, splitter_parallelism, counter_parallelism]
            
        Returns:
            bool: True if rebalance command executed successfully
        """
        try:
            # Map parallelism scheme to component names
            component_parallelism = {
                "sentence-spout": parallelism_scheme[0],  # Source
                "splitter-bolt": parallelism_scheme[1],   # Splitter
                "counter-bolt": parallelism_scheme[2]     # Counter
            }
            
            logger.info(f"Applying rebalance with component parallelisms: {component_parallelism}")
            logger.info(f"Target Nimbus host: {self.nimbus_host}")
            
            # Check if this is local or remote execution
            if self.nimbus_host in ["127.0.0.1", "localhost"]:
                return self._execute_local_rebalance(parallelism_scheme)
            else:
                return self._execute_remote_rebalance(parallelism_scheme)
                
        except Exception as e:
            logger.error(f"Error executing rebalance: {e}")
            return False
    
    def _execute_local_rebalance(self, parallelism_scheme: List[int]) -> bool:
        """Execute Storm rebalance command locally"""
        try:
            # Construct Storm rebalance command for local execution
            rebalance_cmd = [
                "storm", "rebalance", self.topology_name,
                "-w", "10",  # Wait time before rebalancing
                "-n", "90",  # Wait time between each worker assignment
                "-e", f"sentence-spout={parallelism_scheme[0]},splitter-bolt={parallelism_scheme[1]},counter-bolt={parallelism_scheme[2]}"
            ]
            
            logger.info(f"Executing local Storm rebalance: {' '.join(rebalance_cmd)}")
            
            result = subprocess.run(
                rebalance_cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                logger.info("Local Storm rebalance executed successfully")
                logger.debug(f"Output: {result.stdout}")
                return True
            else:
                logger.warning(f"Local rebalance failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Local rebalance error: {e}")
            return False
    
    def _execute_remote_rebalance(self, parallelism_scheme: List[int]) -> bool:
        """Execute Storm rebalance command on remote host via SSH"""
        try:
            # Method 1: SSH command execution
            return self._ssh_rebalance(parallelism_scheme)
            
        except Exception as e:
            logger.error(f"Remote rebalance failed, trying alternative methods: {e}")
            # Method 2: Try Storm REST API as fallback
            return self._rest_api_rebalance(parallelism_scheme)
    
    def _ssh_rebalance(self, parallelism_scheme: List[int]) -> bool:
        """Execute rebalance via SSH to remote Storm cluster"""
        try:
            # Construct remote Storm command
            remote_storm_cmd = f"{self.storm_bin_path} rebalance {self.topology_name} " \
                             f"-e sentence-spout={parallelism_scheme[0]} -e splitter-bolt={parallelism_scheme[1]} -e counter-bolt={parallelism_scheme[2]}"
            
            # SSH command with sshpass for password authentication
            ssh_cmd = [
                "sshpass", "-p", "123",  # Password authentication
                "ssh",
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no",
                f"{self.remote_user}@{self.nimbus_host}",
                remote_storm_cmd
            ]
            
            logger.info(f"Executing SSH rebalance with password authentication")
            logger.debug(f"SSH command: sshpass -p *** ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no {self.remote_user}@{self.nimbus_host} {remote_storm_cmd}")
            
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=120  # Longer timeout for remote execution
            )
            
            if result.returncode == 0:
                logger.info("SSH Storm rebalance executed successfully")
                logger.debug(f"SSH output: {result.stdout}")
                return True
            else:
                logger.warning(f"SSH rebalance failed (code {result.returncode}): {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("SSH rebalance command timed out")
            return False
        except FileNotFoundError:
            logger.error("SSH command not found - make sure SSH is installed")
            return False
        except Exception as e:
            logger.error(f"SSH rebalance error: {e}")
            return False
    
    def _rest_api_rebalance(self, parallelism_scheme: List[int]) -> bool:
        """Execute rebalance via Storm REST API as fallback"""
        try:
            import requests
            
            # Storm REST API endpoint for rebalance
            api_url = f"http://{self.nimbus_host}:8080/api/v1/topology/{self.topology_name}/rebalance"
            
            # Prepare rebalance request
            rebalance_data = {
                "rebalanceOptions": {
                    "numWorkers": None,
                    "executors": {
                        "sentence-spout": parallelism_scheme[0],
                        "splitter-bolt": parallelism_scheme[1], 
                        "counter-bolt": parallelism_scheme[2]
                    }
                }
            }
            
            logger.info(f"Executing REST API rebalance to {api_url}")
            logger.info(f"Rebalance data: {rebalance_data}")
            
            response = requests.post(
                api_url,
                json=rebalance_data,
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                logger.info("REST API rebalance executed successfully")
                logger.debug(f"API response: {response.text}")
                return True
            else:
                logger.warning(f"REST API rebalance failed (status {response.status_code}): {response.text}")
                return False
                
        except ImportError:
            logger.error("requests library not available for REST API rebalance")
            return False
        except Exception as e:
            logger.error(f"REST API rebalance error: {e}")
            return False
    
    def _get_updated_system_metrics(self) -> Tuple[float, float]:
        """
        Get updated system latency and inflight data after rebalance
        
        Returns:
            Tuple[float, float]: (system_latency, inflight_data)
        """
        try:
            # Use the existing method to get spout system metrics
            system_latency, inflight_data = self._get_spout_system_metrics()
            
            if system_latency is None or inflight_data is None:
                logger.warning("Could not get updated system metrics from spouts, using defaults")
                system_latency = 100.0
                inflight_data = 500.0
            
            logger.info(f"Updated system metrics - Latency: {system_latency:.2f}ms, Inflight: {inflight_data:.2f}")
            return system_latency, inflight_data
            
        except Exception as e:
            logger.error(f"Error getting updated system metrics: {e}")
            return 100.0, 500.0

    def start_system_monitoring(self):
        """启动系统状态监控线程"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(
                target=self._system_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("System state monitoring started with 30-second intervals")

    def stop_system_monitoring(self):
        """停止系统状态监控线程"""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
            logger.info("System state monitoring stopped")

    def _system_monitoring_loop(self):
        """系统状态监控循环，每30秒采集一次系统状态"""
        while self.monitoring_active:
            try:
                # 获取当前系统状态
                operators_info, dependencies, system_latency, inflight_data, total_instances = self.get_state()

                # 获取当前并行度方案（从operators_info中提取）
                current_parallelism = []
                # 按operator_id排序以确保顺序正确
                sorted_operators = sorted(operators_info, key=lambda x: x.operator_id)
                for op in sorted_operators:
                    current_parallelism.append(op.instance_count)

                # 计算当前状态的reward
                reward = calculate_reward(
                    L=system_latency,
                    D=inflight_data,
                    parallelism_scheme=current_parallelism,
                    **self.reward_params
                )

                # 创建系统状态记录
                system_state = {
                    'timestamp': time.time(),
                    'operators_info': operators_info,
                    'dependencies': dependencies,
                    'current_parallelism': current_parallelism,
                    'system_latency': system_latency,
                    'inflight_data': inflight_data,
                    'total_instances': total_instances,
                    'reward': reward
                }

                # 线程安全地存储系统状态
                with self.monitoring_lock:
                    self.system_state_buffer.append(system_state)

                
                # 等待30秒
                for _ in range(30):
                    if not self.monitoring_active:
                        break
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Error in system monitoring loop: {e}")
                time.sleep(30)  # 发生错误时等待30秒再继续

    def get_system_state_buffer(self) -> List[Dict]:
        """获取系统状态缓冲区的数据（线程安全）"""
        with self.monitoring_lock:
            return list(self.system_state_buffer)

    def store_system_state(self, operators_info, dependencies, action, reward):
        """
        存储系统状态到system_state_buffer
        这个方法可以被agent调用来手动存储特定的系统状态
        """
        system_state = {
            'timestamp': time.time(),
            'operators_info': operators_info,
            'dependencies': dependencies,
            'action': action,
            'reward': reward
        }

        with self.monitoring_lock:
            self.system_state_buffer.append(system_state)

        logger.debug(f"Manually stored system state: action={action}, reward={reward:.6f}")

    def close(self):
        """Close Redis connection"""
        if self.redis_client:
            try:
                self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.error(f"Failed to close Redis connection: {e}")


def LaStream_Agent_WC(deepseek_api_key: str, num_episodes: int = 10):
    """Test the real environment"""
    logger.info("=== WordCount Real Environment Agent Starts ===")
    
    try:
        # Create environment
        env = WordCountRealEnvironment()

        # Configure for remote Storm cluster (Ubuntu host)
        env.configure_remote_storm(
            nimbus_host="192.168.103.100",
            nimbus_port=6627,
            topology_name="Mydemo",
            remote_user="root",  # Make sure this user exists and has Storm access
            storm_bin_path="/opt/module/storm-1.2.4/bin/storm"  # Adjust path as needed
        )

        # Initialize RL agent
        agent = StreamProcessingRLAgent(
            deepseek_api_key=deepseek_api_key,
            gcn_config={
                'input_dim': 64,
                'hidden_dims': [128, 256, 128],
                'output_dim': 64,
                'dropout': 0.1
            },
            action_evaluator_config={
                'hidden_dims': [256, 512, 256, 128],
                'dropout': 0.2,
                'learning_rate': 0.001
            },
            deepseek_config={
                'max_parallelism': 8,
                'model_name': 'deepseek-chat'
            }
        )
        
        # Training loop
        episode_rewards = []
        episode_metrics = []

        modelcount =0
        
        for episode in range(num_episodes):
            logger.info(f"\n=== Episode {episode + 1}/{num_episodes} ===")

            # 定时器，睡眠120秒,进行采集信息
            if episode > 0:
                logger.info("Waiting 120 seconds before next episode...")
                time.sleep(300)

            # Get state
            operators_info, dependencies, system_latency, inflight_data, total_instances = env.get_state()

            # Calculate average resource utilization
            total_cpu = sum(op.cpu_usage for op in operators_info)
            total_memory = sum(op.memory_usage for op in operators_info)
            avg_cpu = total_cpu / total_instances if total_instances > 0 else 0
            avg_memory = total_memory / total_instances if total_instances > 0 else 0

            # Save resource utilization to file
            with open("resource", "a", encoding="utf-8") as f:
                f.write(f"Average CPU: {avg_cpu:.2f}, Average Memory: {avg_memory:.2f}, Total Instances: {total_instances}, Average latency:{system_latency}, parallelism scheme: {env.operator_instance_counts}\n")



            # 智能体选择动作
            parallelism_scheme, action_info = agent.act(
                operators_info=operators_info,
                dependencies=dependencies,
                num_candidate_schemes=10
            )

            # 检查选择的并行度方案是否与当前配置相同
            current_config = [
                env.operator_instance_counts["sentence"],
                env.operator_instance_counts["splitter"],
                env.operator_instance_counts["counter"]
            ]

            modelcount = modelcount + 1

            if parallelism_scheme == current_config:
                logger.info(f"选择的并行度方案 {parallelism_scheme} 与当前配置相同，跳过本次执行")
                continue

            logger.info(f"选择的并行度方案: {parallelism_scheme}")
            logger.info(f"预测价值: {action_info['predicted_value']:.4f}")

            # 在环境中执行动作
            reward, performance_metrics = env.step(parallelism_scheme)

            # 记录和显示结果
            # logger.info(f"奖励值: {reward:.6f}")
            # logger.info(f"系统延迟: {performance_metrics['system_latency']:.2f}ms")
            # logger.info(f"堆积数据: {performance_metrics['inflight_data']:.2f}")
            # logger.info(f"总实例数: {performance_metrics['total_instances']}")
            # logger.info(f"Rebalance成功: {performance_metrics['rebalance_success']}")
            
            if 'reward_components' in performance_metrics:
                components = performance_metrics['reward_components']
                # logger.info(f"奖励组件: 延迟项={components['latency_term']:.4f}, "
                #            f"数据项={components['data_term']:.4f}, "
                #            f"并行度项={components['parallelism_term']:.4f}")

            # 存储经验
            agent.store_experience(
                operators_info=operators_info,
                dependencies=dependencies,
                action=parallelism_scheme,
                reward=reward
            )

            # Update agent metrics
            agent.update_metrics(reward)

            # Store episode results
            episode_rewards.append(reward)
            episode_metrics.append(performance_metrics)

            
            # Train evaluator every few episodes
            if modelcount > 0 and modelcount % 3 == 0:
                logger.info("Training action evaluator...")
                # 传递environment的system_state_buffer数据给agent训练
                env_system_states = env.get_system_state_buffer()
                training_results = agent.train_evaluator(
                    batch_size=3,
                    num_epochs=50,
                    system_state_buffer=env_system_states
                )
                # logger.info(f"Training results: {training_results}")
                # Save training results to file
                with open("training_results", "a", encoding="utf-8") as f:
                    f.write(f"Training results: {training_results}\n")
                logger.info(f"Training results: {training_results}")
                # Save model
                model_path = "wordcount_real_rl_model.pt"
                agent.save_model(model_path)
                logger.info(f"Model saved to {model_path}")
        
        # 最终统计
        logger.info(f"\n=== 最终结果 ===")
        logger.info(f"平均奖励: {np.mean(episode_rewards):.6f}")
        logger.info(f"最佳奖励: {np.max(episode_rewards):.6f}")
        logger.info(f"奖励改进: {episode_rewards[-1] - episode_rewards[0]:.6f}")

        # Print training statistics
        training_stats = agent.get_training_stats()
        logger.info(f"Training statistics: {training_stats}")

        
        env.close()
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":

     # For the full example, you need to provide your DeepSeek API key
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY","your key")
    
    if deepseek_api_key:
        # Run full example
        results = LaStream_Agent_WC(deepseek_api_key, num_episodes=10000)

    else:
        logger.warning("DEEPSEEK_API_KEY not found in environment variables.")
        logger.info("To run the full example, set your DeepSeek API key:")
        logger.info("export DEEPSEEK_API_KEY='your-api-key-here'")
        logger.info("For now, running GCN embedding test only.")
