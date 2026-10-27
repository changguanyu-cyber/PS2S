from datetime import datetime
import logging
import os
from pathlib import Path
LOG_DIR = Path(os.getenv('PS2S_LOG_DIR', 'ps2s_logs'))
LOG_LEVELS = {0: logging.INFO, 1: logging.DEBUG, -1: logging.WARNING, -2: logging.ERROR}
NOW = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
PS2S_TRAIN_DIR = Path(os.getenv('PS2S_TRAIN_DIR', 'ps2s_training'))
