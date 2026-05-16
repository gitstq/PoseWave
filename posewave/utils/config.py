"""
Configuration Utility
Load and save configuration files.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict, is_dataclass
import logging

logger = logging.getLogger(__name__)


def load_config(
    config_path: Union[str, Path],
    default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Load configuration from file.
    
    Supports JSON and YAML formats.
    
    Args:
        config_path: Path to configuration file
        default: Default configuration if file doesn't exist
    
    Returns:
        Configuration dictionary
    
    Example:
        >>> config = load_config("config.yaml")
        >>> print(config['sample_rate'])
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        if default is not None:
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return default
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if suffix in ['.json']:
                config = json.load(f)
            elif suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        if default is not None:
            return default
        raise


def save_config(
    config: Union[Dict[str, Any], object],
    config_path: Union[str, Path]
) -> None:
    """
    Save configuration to file.
    
    Supports JSON and YAML formats. Dataclass objects are
    automatically converted to dictionaries.
    
    Args:
        config: Configuration dictionary or dataclass
        config_path: Path to save configuration
    
    Example:
        >>> save_config({"sample_rate": 100}, "config.json")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict
    if is_dataclass(config):
        config = asdict(config)
    
    suffix = config_path.suffix.lower()
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if suffix in ['.json']:
                json.dump(config, f, indent=2, ensure_ascii=False)
            elif suffix in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config format: {suffix}")
        
        logger.info(f"Saved configuration to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        raise


def merge_configs(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Values in override take precedence over base.
    
    Args:
        base: Base configuration
        override: Override configuration
    
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result 
            and isinstance(result[key], dict) 
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


class ConfigManager:
    """
    Configuration manager with auto-save and validation.
    
    Example:
        >>> manager = ConfigManager("config.yaml")
        >>> manager.set("sample_rate", 200)
        >>> rate = manager.get("sample_rate")
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        defaults: Optional[Dict[str, Any]] = None,
        auto_save: bool = True
    ):
        self.config_path = Path(config_path)
        self.auto_save = auto_save
        self._config = load_config(config_path, defaults or {})
    
    def get(
        self, 
        key: str, 
        default: Any = None
    ) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(
        self, 
        key: str, 
        value: Any
    ) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        
        if self.auto_save:
            self.save()
    
    def save(self) -> None:
        """Save configuration to file."""
        save_config(self._config, self.config_path)
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = load_config(self.config_path)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration."""
        return self._config.copy()
