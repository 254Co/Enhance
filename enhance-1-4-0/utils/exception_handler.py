from functools import wraps
from .logger import get_logger, log_error

logger = get_logger(__name__)

def handle_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_error(logger, e)
            logger.error(f"Exception occurred in function {func.__name__}: {str(e)}", exc_info=True)
            raise e
    return wrapper