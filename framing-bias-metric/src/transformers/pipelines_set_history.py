def set_history(self, history: List[int]):
    """
        Updates the value of the history of the conversation. The history is represented by a list of :obj:`token_ids`.
        The history is used by the model to generate responses based on the previous conversation turns.

        Args:
            history (:obj:`List[int]`): History of tokens provided and generated for this conversation.
        """
    self.history = history
