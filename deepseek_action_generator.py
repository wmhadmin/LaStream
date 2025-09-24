import openai
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging


class DeepSeekActionGenerator:
    """
    DeepSeek-based action generator for streaming application parallelism optimization.
    Generates parallelism configurations based on operator characteristics.
    """
    
    def __init__(self, 
                 api_key: str,
                 model_name: str = "deepseek-chat",
                 base_url: str = "https://api.deepseek.com",
                 max_parallelism: int = 10):
        """
        Initialize DeepSeek action generator.
        
        Args:
            api_key: DeepSeek API key
            model_name: Model name to use
            base_url: DeepSeek API base URL
            max_parallelism: Maximum parallelism degree for any operator
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.max_parallelism = max_parallelism
        self.logger = logging.getLogger(__name__)
    
    def generate_parallelism_actions(self, 
                                   operators_info: List[Dict],
                                   num_schemes: int = 10) -> List[List[int]]:
        """
        Generate parallelism schemes using DeepSeek.
        
        Args:
            operators_info: List of operator information containing:
                - name: Operator name
                - processing_capability: Average processing capability (0-1)
                - input_rate: Total input rate (normalized)
                - resource_load: Current resource utilization (0-1)
                - dependencies: List of dependent operator indices
            num_schemes: Number of parallelism schemes to generate
             
        Returns:
            List[List[int]]: List of parallelism schemes, each containing parallelism degrees
        """
        try:
            prompt = self._create_parallelism_prompt(operators_info, num_schemes)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in distributed stream processing optimization. Generate parallelism configurations for streaming applications based on operator characteristics."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            schemes = self._parse_parallelism_response(content, len(operators_info))
            
            # Ensure we have the requested number of schemes
            while len(schemes) < num_schemes:
                schemes.extend(self._generate_fallback_schemes(operators_info, num_schemes - len(schemes)))
            
            return schemes[:num_schemes]
            
        except Exception as e:
            self.logger.error(f"Failed to generate parallelism actions: {e}")
            return self._generate_fallback_schemes(operators_info, num_schemes)
    
    def _create_parallelism_prompt(self, operators_info: List[Dict], num_schemes: int) -> str:
        """Create prompt for DeepSeek parallelism generation."""
        
        operators_desc = []
        for i, op in enumerate(operators_info):
            desc = f"""
Operator {i} ({op.operator_name}):
- Processing Capability: {op.processing_capacity} (The maximum data processing rate of operator instances)
- Input Rate: {op.processing_capacity} (The input data rate of the operator )
- CPU Utilization: {op.cpu_usage} 
- Memory Utilization: {op.memory_usage}
- Dependencies: {op.dependencies}
            """
            if op.component_type == "spout":
                desc += f"""\n- Note: This is a spout operator that generates data streams. Its parallelism is set to {op.instance_count} and remains unchanged."""
            
            operators_desc.append(desc.strip())
        
        prompt = f"""
Given a streaming application with {len(operators_info)} operators, generate {num_schemes} different parallelism configuration schemes.

Operators Information:
{chr(10).join(operators_desc)}

Requirements:
1. Each scheme should specify parallelism degree (1-{self.max_parallelism}) for each operator
2. Consider operator characteristics:
    - The processing rate of each operator in streaming applications must keep pace with the input rate.
    -The parallelism of each operator should be set within the range of its specified maximum and minimum values.
    - The configuration of parallelism needs to consider the dependencies between operators to minimize the latency of streaming applications.
    - High resource load operators should be scaled carefully to avoid overloading the system.
    -Under the premise of ensuring system performance, the total number of instances is minimized

3. Generate diverse schemes with different optimization strategies
4. Output format should be JSON array of arrays, e.g.: [[2,4,2,1], [1,8,4,2], ...]

Generate {num_schemes} parallelism schemes:
        """
        
        return prompt.strip()
    
    def _parse_parallelism_response(self, response: str, num_operators: int) -> List[List[int]]:
        """Parse DeepSeek response to extract parallelism schemes."""
        schemes = []
        
        try:
            # Try to find JSON array in response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                parsed_schemes = json.loads(json_str)
                
                for scheme in parsed_schemes:
                    if isinstance(scheme, list) and len(scheme) == num_operators:
                        # Validate and clamp parallelism values
                        valid_scheme = []
                        for p in scheme:
                            if isinstance(p, (int, float)):
                                valid_scheme.append(max(1, min(int(p), self.max_parallelism)))
                            else:
                                valid_scheme.append(1)
                        schemes.append(valid_scheme)
                        
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.warning(f"Failed to parse JSON response: {e}")
            
        # Try to extract schemes from text format
        if not schemes:
            schemes = self._extract_schemes_from_text(response, num_operators)
            
        return schemes
    
    def _extract_schemes_from_text(self, response: str, num_operators: int) -> List[List[int]]:
        """Extract schemes from text format when JSON parsing fails."""
        schemes = []
        lines = response.split('\n')
        
        for line in lines:
            # Look for lines with numbers that could be parallelism values
            if any(char.isdigit() for char in line):
                numbers = []
                current_num = ""
                
                for char in line:
                    if char.isdigit():
                        current_num += char
                    else:
                        if current_num:
                            numbers.append(int(current_num))
                            current_num = ""
                
                if current_num:
                    numbers.append(int(current_num))
                
                if len(numbers) == num_operators:
                    # Validate parallelism values
                    scheme = [max(1, min(p, self.max_parallelism)) for p in numbers]
                    schemes.append(scheme)
        
        return schemes
    
    def _generate_fallback_schemes(self, operators_info: List[Dict], num_schemes: int) -> List[List[int]]:
        """Generate fallback parallelism schemes when DeepSeek fails."""
        schemes = []
        num_operators = len(operators_info)
        
        # Strategy 1: Uniform low parallelism
        schemes.append([2] * num_operators)
        
        # Strategy 2: High parallelism for high-load operators
        high_load_scheme = []
        for op in operators_info:
            resource_load = op.get('resource_load', 0.5)
            input_rate = op.get('input_rate', 0.5)
            
            if resource_load > 0.7 or input_rate > 0.7:
                high_load_scheme.append(min(8, self.max_parallelism))
            else:
                high_load_scheme.append(2)
        schemes.append(high_load_scheme)
        
        # Strategy 3: Inverse of processing capability
        capability_scheme = []
        for op in operators_info:
            capability = op.get('processing_capability', 0.5)
            parallelism = max(1, min(int((1 - capability) * 8 + 1), self.max_parallelism))
            capability_scheme.append(parallelism)
        schemes.append(capability_scheme)
        
        # Strategy 4-10: Random variations
        np.random.seed(42)
        while len(schemes) < num_schemes:
            random_scheme = []
            for op in operators_info:
                base_parallelism = max(1, int(np.random.normal(4, 2)))
                parallelism = max(1, min(base_parallelism, self.max_parallelism))
                random_scheme.append(parallelism)
            schemes.append(random_scheme)
        
        return schemes[:num_schemes]
    
    def get_action_space_size(self, num_operators: int) -> int:
        """Get the size of action space."""
        return self.max_parallelism ** num_operators
    
    def encode_action(self, parallelism_scheme: List[int]) -> int:
        """Encode parallelism scheme into a single action integer."""
        action = 0
        for i, p in enumerate(parallelism_scheme):
            action += (p - 1) * (self.max_parallelism ** i)
        return action
    
    def decode_action(self, action: int, num_operators: int) -> List[int]:
        """Decode action integer back to parallelism scheme."""
        scheme = []
        for i in range(num_operators):
            parallelism = (action // (self.max_parallelism ** i)) % self.max_parallelism + 1
            scheme.append(parallelism)
        return scheme