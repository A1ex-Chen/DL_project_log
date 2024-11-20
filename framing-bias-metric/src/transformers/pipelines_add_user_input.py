def add_user_input(self, text: str, overwrite: bool=False):
    """
        Add a user input to the conversation for the next round. This populates the internal :obj:`new_user_input`
        field.

        Args:
            text (:obj:`str`): The user input for the next conversation round.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not existing and unprocessed user input should be overwritten when this function is called.
        """
    if self.new_user_input:
        if overwrite:
            logger.warning(
                'User input added while unprocessed input was existing: "{}" was overwritten with: "{}".'
                .format(self.new_user_input, text))
            self.new_user_input = text
        else:
            logger.warning(
                'User input added while unprocessed input was existing: "{}" new input ignored: "{}". Set `overwrite` to True to overwrite unprocessed user input'
                .format(self.new_user_input, text))
    else:
        self.new_user_input = text
