import yaml
from pathlib import Path
import os
from typing import Dict, Any

class ConfigLoader:
    """Configuration loader with environment variable substitution"""
    
    def __init__(self, config_dir="config"):
        self.config_dir = Path(config_dir)
        self.configs = {}
        
    def load_all(self) -> Dict[str, Any]:
        """Load all configuration files"""
        config_files = {
            "text": "text_config.yaml",
            "paths": "paths.yaml",
            "summarization": "summarization_config.yaml"
        }
        
        for name, file_name in config_files.items():
            config_path = self.config_dir / file_name
            if config_path.exists():
                self.configs[name] = self._load_yaml(config_path)
        
        # Merge configurations
        merged_config = {}
        for config in self.configs.values():
            self._deep_update(merged_config, config)
        
        # Substitute environment variables
        self._substitute_env_vars(merged_config)
        
        return merged_config
    
    def _load_yaml(self, path: Path) -> Dict:
        """Load YAML file with environment variable resolution"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace environment variables
        content = self._replace_env_in_content(content)
        
        return yaml.safe_load(content)
    
    def _replace_env_in_content(self, content: str) -> str:
        """Replace ${VAR} with environment variables"""
        import re
        
        def replace_match(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        pattern = r'\$\{([^}]+)\}'
        return re.sub(pattern, replace_match, content)
    
    def _deep_update(self, original: Dict, update: Dict):
        """Deep update dictionary"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _substitute_env_vars(self, config: Dict):
        """Recursively substitute environment variables in config"""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                config[key] = os.environ.get(var_name, value)
    
    def save_config(self, config: Dict, name: str):
        """Save configuration to file"""
        config_path = self.config_dir / f"{name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")

# Singleton instance
config_loader = ConfigLoader()