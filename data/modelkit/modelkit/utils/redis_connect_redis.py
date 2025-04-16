@retry(wait=REDIS_RETRY_POLICY.wait, stop=REDIS_RETRY_POLICY.stop, retry=
    REDIS_RETRY_POLICY.retry, after=REDIS_RETRY_POLICY.after, reraise=
    REDIS_RETRY_POLICY.reraise)
def connect_redis(host, port):
    redis_cache = redis.Redis(host=host, port=port)
    if not redis_cache.ping():
        raise ConnectionError('Cannot connect to redis')
    return redis_cache
