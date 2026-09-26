import logging , sys , colorama
import sys
from datetime import datetime

colorama.just_fix_windows_console()

class ColorFormatter(logging.Formatter):
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    GRAY = '\033[90m'

    # Fixed width for level names
    LEVEL_WIDTH = 8

    def format(self, record):
        # Timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        time_str = f"{self.GRAY}{timestamp}{self.RESET}"

        # Level with color
        color = self.COLORS.get(record.levelname, self.RESET)
        level_str = f"{color}{self.BOLD}{record.levelname:<{self.LEVEL_WIDTH}}{self.RESET}"

        # Logger name
        name_str = f"{self.BOLD}{record.name}{self.RESET}"

        # Location (filename:lineno)
        location = f"{self.GRAY}{record.filename}:{record.lineno}{self.RESET}"

        # Header line
        header = f"{time_str} | {level_str} | {name_str} | {location}"

        # Message
        message = record.getMessage()
        if record.exc_info:
            message += "\n" + self.formatException(record.exc_info)

        # Separator
        separator = f"{self.GRAY}{'─' * 80}{self.RESET}"

        return f"{separator}\n{header}\n{self.DIM}{message}{self.RESET}\n"


def get_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColorFormatter())
        logger.addHandler(handler)

    return logger