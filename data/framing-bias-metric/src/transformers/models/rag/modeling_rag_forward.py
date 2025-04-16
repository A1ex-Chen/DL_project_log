@add_start_docstrings_to_model_forward(RAG_FORWARD_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=RetrievAugLMMarginOutput,
    config_class=_CONFIG_FOR_DOC)
def forward(self, input_ids=None, attention_mask=None, encoder_outputs=None,
    decoder_input_ids=None, decoder_attention_mask=None, past_key_values=
    None, context_input_ids=None, context_attention_mask=None, doc_scores=
    None, use_cache=None, output_attentions=None, output_hidden_states=None,
    output_retrieved=None, do_marginalize=None, reduce_loss=None, labels=
    None, n_docs=None, **kwargs):
    """
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> input_dict = tokenizer.prepare_seq2seq_batch("How many people live in Paris?", "In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = input_dict["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=input_dict["labels"])

            >>> # or use retriever separately
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=input_dict["labels"])

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
    n_docs = n_docs if n_docs is not None else self.config.n_docs
    do_marginalize = (do_marginalize if do_marginalize is not None else
        self.config.do_marginalize)
    reduce_loss = (reduce_loss if reduce_loss is not None else self.config.
        reduce_loss)
    if labels is not None:
        if decoder_input_ids is None:
            decoder_input_ids = labels
        use_cache = False
    outputs = self.rag(input_ids=input_ids, attention_mask=attention_mask,
        encoder_outputs=encoder_outputs, decoder_input_ids=
        decoder_input_ids, decoder_attention_mask=decoder_attention_mask,
        context_input_ids=context_input_ids, context_attention_mask=
        context_attention_mask, doc_scores=doc_scores, past_key_values=
        past_key_values, use_cache=use_cache, output_attentions=
        output_attentions, output_hidden_states=output_hidden_states,
        output_retrieved=output_retrieved, n_docs=n_docs)
    loss = None
    logits = outputs.logits
    if labels is not None:
        assert decoder_input_ids is not None
        loss = self.get_nll(outputs.logits, outputs.doc_scores, labels,
            reduce_loss=reduce_loss, epsilon=self.config.label_smoothing,
            n_docs=n_docs)
    if do_marginalize:
        logits = self.marginalize(logits, outputs.doc_scores, n_docs)
    return RetrievAugLMMarginOutput(loss=loss, logits=logits, doc_scores=
        outputs.doc_scores, past_key_values=outputs.past_key_values,
        context_input_ids=outputs.context_input_ids, context_attention_mask
        =outputs.context_attention_mask, retrieved_doc_embeds=outputs.
        retrieved_doc_embeds, retrieved_doc_ids=outputs.retrieved_doc_ids,
        question_encoder_last_hidden_state=outputs.
        question_encoder_last_hidden_state, question_enc_hidden_states=
        outputs.question_enc_hidden_states, question_enc_attentions=outputs
        .question_enc_attentions, generator_enc_last_hidden_state=outputs.
        generator_enc_last_hidden_state, generator_enc_hidden_states=
        outputs.generator_enc_hidden_states, generator_enc_attentions=
        outputs.generator_enc_attentions, generator_dec_hidden_states=
        outputs.generator_dec_hidden_states, generator_dec_attentions=
        outputs.generator_dec_attentions)
