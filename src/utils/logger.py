# src/utils/logger.py

from loguru import logger
import sys

# Remove default handler and add our own stream handler
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time}</green> | <level>{level}</level> | <cyan>{message}</cyan>")

# Safely add 'SUCCESS' level
try:
    logger.level("SUCCESS", no=25, color="<green>", icon="✅")
except ValueError:
    pass  # Level already exists, ignore

# Optional shortcut
def success(msg, *args, **kwargs):
    logger.log("SUCCESS", msg, *args, **kwargs)

logger.success = success
