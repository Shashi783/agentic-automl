from typing import Dict, Any, Optional
import logging
from .secrets_manager import get_secret, get_config

logger = logging.getLogger(__name__)

# app/core/settings_manager.py
import logging
from functools import lru_cache
from hinting_agent_app.config import Settings
from hinting_agent_app.secrets_manager import SecretsManager

logger = logging.getLogger(__name__)

### FILE: service/settings.py
from __future__ import annotations
from pydantic import BaseSettings, Field

#####
class Settings(BaseSettings):
    # Storage paths (local for demo; swap to S3/Blob in prod)
    work_dir: str = Field(default="/tmp/mlsvc")
    models_dir: str = Field(default="/tmp/mlsvc/models")
    data_dir: str = Field(default="/tmp/mlsvc/data")

    # Service
    host: str = "0.0.0.0"
    port: int = 8080

    class Config:
        env_prefix = "MLSVC_"

#####


class SettingsManager:
    """
    Manages app settings and secrets.
    Loads base Settings, injects secrets from SecretsManager,
    and ensures singleton behavior.
    """

    def __init__(self, env_file: str | None = ".env"):
        # Secrets manager instance
        self._secrets_manager = SecretsManager(env_file=env_file)
        self._settings: Settings | None = None

    @property
    def settings(self) -> Settings:
        """
        Returns a singleton Settings instance.
        Loads and injects secrets if not already initialized.
        """
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value.
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        return self._settings.get(key, default)

    def get_secret_value(self, key: str, default: str = None) -> Optional[str]:
        """Get a specific secret value.
        
        Args:
            key: Secret key
            default: Default value if secret not found
            
        Returns:
            Secret value or default
        """
        # if key not in self._secrets:
        #     self._secrets[key] = get_secret(key)
        return self._secrets_manager.get(key, default) 

    def _load_settings(self) -> Settings:
        settings = Settings()

        secret_fields = ["DATABASE_URL", "JWT_SECRET", "ANOTHER_SECRET", "SOME_API_KEY"]
        for field in secret_fields:
            if not getattr(settings, field, None):
                secret_value = self._secrets_manager.get(field)
                if secret_value:
                    setattr(settings, field, secret_value)
                    logger.info(f"Loaded {field} from SecretsManager")

        return settings

    def override(self, **overrides) -> Settings:
        """
        Override settings for testing or special cases.
        Clears the cached instance and rebuilds Settings.
        """
        self._settings = Settings(**overrides)
        return self._settings


# Singleton instance for the app
settings_manager = SettingsManager()
settings = settings_manager.settings

