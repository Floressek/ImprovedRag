import subprocess
import time
from typing import Optional

from src.ragx.utils.settings import settings

import requests
import os

import logging

logger = logging.getLogger(__name__)




def ensure_ollama_running(timeout: int, host: Optional[str] = None) -> bool:
    """Check if Ollama is running, start it if not."""
    host = host or settings.llm.ollama_host
    timeout = timeout or 30
    try:
        response = requests.get(f"{host}/api/tags", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass

    try:
        env = os.environ.copy()
        env['OLLAMA_MODELS'] = settings.llm.ollama_models_path
        logger.info(f"Using OLLAMA_MODELS path: {env['OLLAMA_MODELS']}")

        if os.name == 'nt':
            logger.info("Starting Ollama on Windows...")
            subprocess.Popen(['ollama', 'serve'],
                             creationflags=subprocess.CREATE_NEW_CONSOLE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        else:
            logger.info("Starting Ollama on Unix...")
            subprocess.Popen(['ollama', 'serve'],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)

        for _ in range(timeout):
            time.sleep(1)
            try:
                response = requests.get(f"{host}/api/tags", timeout=2)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                continue

        logger.error("Ollama failed to start within timeout")
        return False
    except FileNotFoundError:
        logger.error("'ollama' command not found in PATH")
        return False
    except Exception as e:
        logger.error(f"Failed to start Ollama: {e}")
        raise RuntimeError(f"Failed to start Ollama server: {e}")
