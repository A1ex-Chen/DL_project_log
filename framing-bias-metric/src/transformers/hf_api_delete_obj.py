def delete_obj(self, token: str, filename: str, organization: Optional[str]
    =None):
    """
        HuggingFace S3-based system, used for datasets and metrics.

        Call HF API to delete a file stored by user
        """
    path = '{}/api/datasets/deleteObj'.format(self.endpoint)
    r = requests.delete(path, headers={'authorization': 'Bearer {}'.format(
        token)}, json={'filename': filename, 'organization': organization})
    r.raise_for_status()
