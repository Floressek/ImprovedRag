from __future__ import annotations
import logging
import sys
import warnings

warnings.filterwarnings("ignore", message=".*flash_attn is not installed.*")


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{self.BOLD}{record.levelname:8}{self.RESET}"
        record.asctime = self.formatTime(record, '%H:%M:%S')

        # Shorten logger name
        name_parts = record.name.split('.')
        if len(name_parts) > 2:
            record.name = f"{name_parts[0]}...{name_parts[-1]}"
        elif len(name_parts) == 2:
            record.name = name_parts[-1]

        return super().format(record)


def setup_logging(level: str = "INFO") -> None:
    """Setup structured logging with colors."""
    if logging.getLogger().handlers:
        return

    # Create handler
    handler = logging.StreamHandler(sys.stdout)
    formatter = ColoredFormatter(
        fmt=f"%(asctime)s {ColoredFormatter.COLORS['INFO']}║{ColoredFormatter.RESET} %(levelname)s {ColoredFormatter.COLORS['INFO']}║{ColoredFormatter.RESET} %(name)-15s {ColoredFormatter.COLORS['INFO']}║{ColoredFormatter.RESET} %(message)s"
    )
    handler.setFormatter(formatter)

    # Setup root logger
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(level.upper() if isinstance(level, str) else level)

    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

    logging.getLogger("src.ragx").setLevel(logging.INFO)
    logging.getLogger("__main__").setLevel(logging.INFO)
