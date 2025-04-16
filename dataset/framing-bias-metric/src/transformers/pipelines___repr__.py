def __repr__(self):
    """
        Generates a string representation of the conversation.

        Return:
            :obj:`str`:

            Example: Conversation id: 7d15686b-dc94-49f2-9c4b-c9eac6a1f114 user >> Going to the movies tonight - any
            suggestions? bot >> The Big Lebowski
        """
    output = 'Conversation id: {} \n'.format(self.uuid)
    for user_input, generated_response in zip(self.past_user_inputs, self.
        generated_responses):
        output += 'user >> {} \n'.format(user_input)
        output += 'bot >> {} \n'.format(generated_response)
    if self.new_user_input is not None:
        output += 'user >> {} \n'.format(self.new_user_input)
    return output
