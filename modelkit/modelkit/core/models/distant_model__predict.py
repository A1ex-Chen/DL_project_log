@retry(wait=SERVICE_MODEL_RETRY_POLICY.wait, stop=
    SERVICE_MODEL_RETRY_POLICY.stop, retry=SERVICE_MODEL_RETRY_POLICY.retry,
    after=SERVICE_MODEL_RETRY_POLICY.after, reraise=
    SERVICE_MODEL_RETRY_POLICY.reraise)
def _predict(self, item, **kwargs):
    if not self.requests_session:
        self.requests_session = requests.Session()
    try:
        item = json.dumps(item)
    except TypeError:
        item = item.model_dump_json()
    response = self.requests_session.post(self.endpoint, params=kwargs.get(
        'endpoint_params', self.endpoint_params), data=item, headers={
        'content-type': 'application/json', **kwargs.get('endpoint_headers',
        self.endpoint_headers)}, timeout=self.timeout)
    if response.status_code != 200:
        raise DistantHTTPModelError(response.status_code, response.reason,
            response.text)
    return response.json()
