@add_start_docstrings_to_model_forward(DISTILBERT_INPUTS_DOCSTRING.format(
    'batch_size, num_choices, sequence_length'))
@replace_return_docstrings(output_type=MultipleChoiceModelOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, head_mask=None,
    inputs_embeds=None, labels=None, output_attentions=None,
    output_hidden_states=None, return_dict=None):
    """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)

        Returns:

        Examples::

            >>> from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
            >>> import torch

            >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            >>> model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> choice0 = "It is eaten with a fork and a knife."
            >>> choice1 = "It is eaten while held in the hand."
            >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
            >>> outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) # batch size is 1

            >>> # the linear classifier still needs to be trained
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
    return_dict = (return_dict if return_dict is not None else self.config.
        use_return_dict)
    num_choices = input_ids.shape[1
        ] if input_ids is not None else inputs_embeds.shape[1]
    input_ids = input_ids.view(-1, input_ids.size(-1)
        ) if input_ids is not None else None
    attention_mask = attention_mask.view(-1, attention_mask.size(-1)
        ) if attention_mask is not None else None
    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2),
        inputs_embeds.size(-1)) if inputs_embeds is not None else None
    outputs = self.distilbert(input_ids, attention_mask=attention_mask,
        head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions
        =output_attentions, output_hidden_states=output_hidden_states,
        return_dict=return_dict)
    hidden_state = outputs[0]
    pooled_output = hidden_state[:, 0]
    pooled_output = self.pre_classifier(pooled_output)
    pooled_output = nn.ReLU()(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    reshaped_logits = logits.view(-1, num_choices)
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(reshaped_logits, labels)
    if not return_dict:
        output = (reshaped_logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output
    return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits,
        hidden_states=outputs.hidden_states, attentions=outputs.attentions)
