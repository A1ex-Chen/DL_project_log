@staticmethod
def _iterate_content(response: requests.Response) ->None:
    """
        Process the streamed HTTP response data.

        Args:
            response (requests.Response): The response object from the file download request.

        Returns:
            None
        """
    for _ in response.iter_content(chunk_size=1024):
        pass
