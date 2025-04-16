def presign(self, token: str, filename: str, organization: Optional[str]=None
    ) ->PresignedUrl:
    """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to get a presigned url to upload `filename` to S3.
        """
    path = '{}/api/datasets/presign'.format(self.endpoint)
    r = requests.post(path, headers={'authorization': 'Bearer {}'.format(
        token)}, json={'filename': filename, 'organization': organization})
    r.raise_for_status()
    d = r.json()
    return PresignedUrl(**d)
