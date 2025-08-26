import os
import json
from typing import Optional
import logging
from pathlib import Path
from hinting_agent_app.config import get_settings, Settings

logger = logging.getLogger(__name__)

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    boto3 = None

from dotenv import dotenv_values


class SecretsManager:
    def __init__(self, env_file: str | None = None):
        """
        env_file: Optional path to a .env file to load secrets from
        """
        self.env_file = env_file
        self.env_vars = dotenv_values(env_file) if env_file and os.path.exists(env_file) else {}

    def get(self, secret_name: str) -> Optional[str]:
        """
        Get a secret value from environment variables, .env file, or AWS Secrets Manager.
        """
        try:
            # 1️⃣ Environment variables
            if secret_name in os.environ:
                logger.warning(f"Secret {secret_name} not found in environment")
                return os.environ[secret_name]

            # 2️⃣ .env file
            if secret_name in self.env_vars:
                logger.warning(f"Secret {secret_name} not found in env_file")
                return self.env_vars[secret_name]

            # 3️⃣ AWS Secrets Manager
            if boto3:
                client = boto3.client("secretsmanager")
                try:
                    response = client.get_secret_value(SecretId=secret_name)
                    secret = response.get("SecretString")
                    if secret:
                        logger.warning(f"Secret {secret_name} not found in AWS Secrets Manager")
                        return secret
                except ClientError as e:
                    logger.warning(f"Secret {secret_name} not found in AWS Secrets Manager: {e}")

            logger.warning(f"Secret {secret_name} not found in environment, env_file, or AWS Secrets Manager")
            return None

        except Exception as e:
            logger.error(f"Error retrieving secret {secret_name}: {e}")
            return None




# logger = logging.getLogger(__name__)

# def get_secret(secret_name: str) -> Optional[str]:
#     """Get a secret value from environment variables or secrets file.
    
#     Args:
#         secret_name: Name of the secret to retrieve
        
#     Returns:
#         The secret value if found, None otherwise
#     """
#     try:
#         # First try environment variables
#         if secret_name in os.environ:
#             return os.environ[secret_name]

#         # Then try secrets file
#         secrets_file = os.path.join(os.path.dirname(__file__), "..", "..", "secrets.json")
#         if os.path.exists(secrets_file):
#             with open(secrets_file, "r") as f:
#                 secrets = json.load(f)
#                 return secrets.get(secret_name)

#         logger.warning(f"Secret {secret_name} not found in environment or secrets file")
#         return None

#     except Exception as e:
#         logger.error(f"Error retrieving secret {secret_name}: {e}")
#         return None


