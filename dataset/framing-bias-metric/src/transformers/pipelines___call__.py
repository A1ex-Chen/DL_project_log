def __call__(self, conversations: Union[Conversation, List[Conversation]],
    clean_up_tokenization_spaces=True, **generate_kwargs):
    """
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`):
                Conversations to generate responses for.
            clean_up_tokenization_spaces (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework `here <./model.html#generative-models>`__).

        Returns:
            :class:`~transformers.Conversation` or a list of :class:`~transformers.Conversation`: Conversation(s) with
            updated generated responses for those containing a new user input.
        """
    if isinstance(conversations, Conversation):
        conversations = [conversations]
    if isinstance(conversations, list):
        for conversation in conversations:
            assert isinstance(conversation, Conversation
                ), 'DialoguePipeline expects a Conversation or list of Conversations as an input'
            if conversation.new_user_input is None:
                raise ValueError(
                    "Conversation with UUID {} does not contain new user input to process. Add user inputs with the conversation's `add_user_input` method"
                    .format(type(conversation.uuid)))
        assert self.tokenizer.pad_token_id is not None or self.tokenizer.eos_token_id is not None, 'Please make sure that the tokenizer has a pad_token_id or eos_token_id when using a batch input'
    else:
        raise ValueError(
            'DialoguePipeline expects a Conversation or list of Conversations as an input'
            )
    with self.device_placement():
        inputs = self._parse_and_tokenize([conversation.new_user_input for
            conversation in conversations])
        histories = [conversation.history for conversation in conversations]
        max_length = generate_kwargs.get('max_length', self.model.config.
            max_length)
        inputs = self._concat_inputs_history(inputs, histories, max_length)
        if self.framework == 'pt':
            inputs = self.ensure_tensor_on_device(**inputs)
            input_length = inputs['input_ids'].shape[-1]
        elif self.framework == 'tf':
            input_length = tf.shape(inputs['input_ids'])[-1].numpy()
        if input_length > 0.9 * max_length:
            logger.warning(
                'Longest conversation length: {} is bigger than 0.9 * max_length: {}. You might consider trimming the early phase of the conversation'
                .format(input_length, max_length))
        generated_responses = self.model.generate(inputs['input_ids'],
            attention_mask=inputs['attention_mask'], **generate_kwargs)
        if self.model.config.is_encoder_decoder:
            if self.framework == 'pt':
                history = torch.cat((inputs['input_ids'],
                    generated_responses[:, 1:]), 1)
            elif self.framework == 'tf':
                history = tf.concat([inputs['input_ids'],
                    generated_responses[:, 1:]], 1)
        else:
            history = generated_responses
        history = self._clean_padding_history(history)
        if self.model.config.is_encoder_decoder:
            start_position = 1
        else:
            start_position = input_length
        output = []
        for conversation_index, conversation in enumerate(conversations):
            conversation.mark_processed()
            conversation.generated_responses.append(self.tokenizer.decode(
                generated_responses[conversation_index][start_position:],
                skip_special_tokens=True, clean_up_tokenization_spaces=
                clean_up_tokenization_spaces))
            conversation.set_history(history[conversation_index])
            output.append(conversation)
        if len(output) == 1:
            return output[0]
        else:
            return output
