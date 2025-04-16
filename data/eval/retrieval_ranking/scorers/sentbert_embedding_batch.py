def embedding_batch(self, list_inputText, contextual=False):
    list_inputText = [x.lower().strip() for x in list_inputText]
    output_value = ('sentence_embedding' if not contextual else
        'token_embeddings')
    outputs = self.model.encode(list_inputText, convert_to_tensor=True,
        output_value=output_value, show_progress_bar=False)
    if output_value == 'sentence_embedding':
        return outputs.cpu()
    outputs = [output.cpu() for output in outputs]
    return outputs
