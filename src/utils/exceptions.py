class MaskDetectionError(Exception):
    """Base class for project errors."""

def wrap_error(fn):
    """Decorator: log & re‑raise custom error."""
    from functools import wraps
    from src.utils.logger import logger

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as err:
            logger.exception(f"Error in {fn.__name__}: {err}")
            raise MaskDetectionError(str(err)) from err
    return _wrapper
