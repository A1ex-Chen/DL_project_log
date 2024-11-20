def append_response(self, response: str):
    """
        Append a response to the list of generated responses.

        Args:
            response (:obj:`str`): The model generated response.
        """
    self.generated_responses.append(response)
