import requests
from structlog import get_logger
from tenacity import retry_if_exception, stop_after_attempt, wait_random_exponential

logger = get_logger(__name__)





    else:


    return {
        "wait": wait_random_exponential(multiplier=1, min=4, max=10),
        "stop": stop_after_attempt(5),
        "retry": retry_if_exception(is_retry_eligible),
        "after": log_after_retry,
        "reraise": True,
    }