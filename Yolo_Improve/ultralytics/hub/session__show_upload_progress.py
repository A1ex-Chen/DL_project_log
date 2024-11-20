@staticmethod
def _show_upload_progress(content_length: int, response: requests.Response
    ) ->None:
    """
        Display a progress bar to track the upload progress of a file download.

        Args:
            content_length (int): The total size of the content to be downloaded in bytes.
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
    with TQDM(total=content_length, unit='B', unit_scale=True, unit_divisor
        =1024) as pbar:
        for data in response.iter_content(chunk_size=1024):
            pbar.update(len(data))
