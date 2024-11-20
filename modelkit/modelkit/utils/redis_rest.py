from dataclasses import dataclass
from typing import Callable

from structlog import get_logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

logger = get_logger(__name__)
try:
    import redis
except ImportError:  # pragma: no cover
    logger.debug("Redis is not available " "(install modelkit[redis] or redis)")


class RedisCacheException(Exception):
    pass






@dataclass
class REDIS_RETRY_POLICY:
    wait: wait_random_exponential = wait_random_exponential(multiplier=1, min=4, max=10)
    stop: stop_after_attempt = stop_after_attempt(5)
    retry: retry_if_exception = retry_if_exception(retriable_error)
    after: Callable = log_after_retry
    reraise: bool = True


@retry(
    wait=REDIS_RETRY_POLICY.wait,
    stop=REDIS_RETRY_POLICY.stop,
    retry=REDIS_RETRY_POLICY.retry,
    after=REDIS_RETRY_POLICY.after,
    reraise=REDIS_RETRY_POLICY.reraise,
)