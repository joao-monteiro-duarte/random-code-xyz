import os
import logging

logger = logging.getLogger(__name__)

def configure_openrouter_env(api_key: str, service_name: str, debug: bool = False):
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
    if debug:
        logger.info(f"{service_name}: Configured OpenRouter with API key ending in {api_key[-4:]}")