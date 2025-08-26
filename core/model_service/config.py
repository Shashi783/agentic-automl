import os
import yaml
import logging
from pydantic import BaseSettings, AnyUrl
from functools import lru_cache

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    APP_NAME: str = "model-service-api"
    ENV: str = "prod"
    JWT_SECRET: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MIN: int = 15
    REFRESH_TOKEN_TTL_DAYS: int = 7
    CORS_ORIGINS: list[str] = ["https://your-frontend.example.com"]
    DATABASE_URL: AnyUrl

    # Cookies
    COOKIE_DOMAIN: str | None = None  # set your domain in prod
    COOKIE_SECURE: bool = True
    COOKIE_SAMESITE: str = "lax"  # "strict" or "none" (requires https)

    # Prompts YAML
    PROMPTS_YAML_PATH: str = "core/model_service/prompts/model_selection.prompt"

    class Config:
        env_file = ".env"
        case_sensitive = True

    def load_prompts_yaml(self) -> dict:
        """Loads and returns the content of the prompts YAML file."""
        yaml_path = os.path.join(os.getcwd(), self.PROMPTS_YAML_PATH)
        try:
            with open(yaml_path, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
            logger.info(f"Successfully loaded prompts from {yaml_path}")
            return data
        except FileNotFoundError:
            logger.error(f"Prompts YAML file not found at {yaml_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing prompts YAML file {yaml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading prompts from {yaml_path}: {e}")
            raise

    def get(self, key: str, default: Any = None) -> Any:
            """
            Dynamically fetch the value of a setting by name.
            
            Args:
                key: Name of the setting attribute.
                default: Value to return if key is not found.
            
            Returns:
                The value of the setting, or default if not present.
            """
            return getattr(self, key, default)

@lru_cache
def get_settings() -> Settings:
    return Settings()
