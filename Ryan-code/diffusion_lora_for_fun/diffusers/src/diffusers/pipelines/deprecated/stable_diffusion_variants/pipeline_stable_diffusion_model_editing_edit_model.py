@torch.no_grad()
def edit_model(self, source_prompt: str, destination_prompt: str, lamb:
    float=0.1, restart_params: bool=True):
    """
        Apply model editing via closed-form solution (see Eq. 5 in the TIME [paper](https://arxiv.org/abs/2303.08084)).

        Args:
            source_prompt (`str`):
                The source prompt containing the concept to be edited.
            destination_prompt (`str`):
                The destination prompt. Must contain all words from `source_prompt` with additional ones to specify the
                target edit.
            lamb (`float`, *optional*, defaults to 0.1):
                The lambda parameter specifying the regularization intesity. Smaller values increase the editing power.
            restart_params (`bool`, *optional*, defaults to True):
                Restart the model parameters to their pre-trained version before editing. This is done to avoid edit
                compounding. When it is `False`, edits accumulate.
        """
    if restart_params:
        num_ca_clip_layers = len(self.ca_clip_layers)
        for idx_, l in enumerate(self.ca_clip_layers):
            l.to_v = copy.deepcopy(self.og_matrices[idx_])
            self.projection_matrices[idx_] = l.to_v
            if self.with_to_k:
                l.to_k = copy.deepcopy(self.og_matrices[num_ca_clip_layers +
                    idx_])
                self.projection_matrices[num_ca_clip_layers + idx_] = l.to_k
    old_texts = [source_prompt]
    new_texts = [destination_prompt]
    base = old_texts[0] if old_texts[0][0:1] != 'A' else 'a' + old_texts[0][1:]
    for aug in self.with_augs:
        old_texts.append(aug + base)
    base = new_texts[0] if new_texts[0][0:1] != 'A' else 'a' + new_texts[0][1:]
    for aug in self.with_augs:
        new_texts.append(aug + base)
    old_embs, new_embs = [], []
    for old_text, new_text in zip(old_texts, new_texts):
        text_input = self.tokenizer([old_text, new_text], padding=
            'max_length', max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.
            device))[0]
        old_emb, new_emb = text_embeddings
        old_embs.append(old_emb)
        new_embs.append(new_emb)
    idxs_replaces = []
    for old_text, new_text in zip(old_texts, new_texts):
        tokens_a = self.tokenizer(old_text).input_ids
        tokens_b = self.tokenizer(new_text).input_ids
        tokens_a = [(self.tokenizer.encode('a ')[1] if self.tokenizer.
            decode(t) == 'an' else t) for t in tokens_a]
        tokens_b = [(self.tokenizer.encode('a ')[1] if self.tokenizer.
            decode(t) == 'an' else t) for t in tokens_b]
        num_orig_tokens = len(tokens_a)
        idxs_replace = []
        j = 0
        for i in range(num_orig_tokens):
            curr_token = tokens_a[i]
            while tokens_b[j] != curr_token:
                j += 1
            idxs_replace.append(j)
            j += 1
        while j < 77:
            idxs_replace.append(j)
            j += 1
        while len(idxs_replace) < 77:
            idxs_replace.append(76)
        idxs_replaces.append(idxs_replace)
    contexts, valuess = [], []
    for old_emb, new_emb, idxs_replace in zip(old_embs, new_embs, idxs_replaces
        ):
        context = old_emb.detach()
        values = []
        with torch.no_grad():
            for layer in self.projection_matrices:
                values.append(layer(new_emb[idxs_replace]).detach())
        contexts.append(context)
        valuess.append(values)
    for layer_num in range(len(self.projection_matrices)):
        mat1 = lamb * self.projection_matrices[layer_num].weight
        mat2 = lamb * torch.eye(self.projection_matrices[layer_num].weight.
            shape[1], device=self.projection_matrices[layer_num].weight.device)
        for context, values in zip(contexts, valuess):
            context_vector = context.reshape(context.shape[0], context.
                shape[1], 1)
            context_vector_T = context.reshape(context.shape[0], 1, context
                .shape[1])
            value_vector = values[layer_num].reshape(values[layer_num].
                shape[0], values[layer_num].shape[1], 1)
            for_mat1 = (value_vector @ context_vector_T).sum(dim=0)
            for_mat2 = (context_vector @ context_vector_T).sum(dim=0)
            mat1 += for_mat1
            mat2 += for_mat2
        self.projection_matrices[layer_num].weight = torch.nn.Parameter(
            mat1 @ torch.inverse(mat2))
