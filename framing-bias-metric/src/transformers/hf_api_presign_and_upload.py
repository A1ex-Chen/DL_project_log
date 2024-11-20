def presign_and_upload(self, token: str, filename: str, filepath: str,
    organization: Optional[str]=None) ->str:
    """
        HuggingFace S3-based system, used for datasets and metrics.

        Get a presigned url, then upload file to S3.

        Outputs: url: Read-only url for the stored file on S3.
        """
    urls = self.presign(token, filename=filename, organization=organization)
    with open(filepath, 'rb') as f:
        pf = TqdmProgressFileReader(f)
        data = f if pf.total_size > 0 else ''
        r = requests.put(urls.write, data=data, headers={'content-type':
            urls.type})
        r.raise_for_status()
        pf.close()
    return urls.access
