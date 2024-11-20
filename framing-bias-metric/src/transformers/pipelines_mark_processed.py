def mark_processed(self):
    """
        Mark the conversation as processed (moves the content of :obj:`new_user_input` to :obj:`past_user_inputs`) and
        empties the :obj:`new_user_input` field.
        """
    if self.new_user_input:
        self.past_user_inputs.append(self.new_user_input)
    self.new_user_input = None
