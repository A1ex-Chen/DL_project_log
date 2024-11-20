def unload_textual_inversion(self, tokens: Optional[Union[str, List[str]]]=
    None, tokenizer: Optional['PreTrainedTokenizer']=None, text_encoder:
    Optional['PreTrainedModel']=None):
    """
        Unload Textual Inversion embeddings from the text encoder of [`StableDiffusionPipeline`]

        Example:
        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained("runwayml/stable-diffusion-v1-5")

        # Example 1
        pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
        pipeline.load_textual_inversion("sd-concepts-library/moeb-style")

        # Remove all token embeddings
        pipeline.unload_textual_inversion()

        # Example 2
        pipeline.load_textual_inversion("sd-concepts-library/moeb-style")
        pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")

        # Remove just one token
        pipeline.unload_textual_inversion("<moe-bius>")

        # Example 3: unload from SDXL
        pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
        embedding_path = hf_hub_download(
            repo_id="linoyts/web_y2k", filename="web_y2k_emb.safetensors", repo_type="model"
        )

        # load embeddings to the text encoders
        state_dict = load_file(embedding_path)

        # load embeddings of text_encoder 1 (CLIP ViT-L/14)
        pipeline.load_textual_inversion(
            state_dict["clip_l"],
            token=["<s0>", "<s1>"],
            text_encoder=pipeline.text_encoder,
            tokenizer=pipeline.tokenizer,
        )
        # load embeddings of text_encoder 2 (CLIP ViT-G/14)
        pipeline.load_textual_inversion(
            state_dict["clip_g"],
            token=["<s0>", "<s1>"],
            text_encoder=pipeline.text_encoder_2,
            tokenizer=pipeline.tokenizer_2,
        )

        # Unload explicitly from both text encoders abd tokenizers
        pipeline.unload_textual_inversion(
            tokens=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer
        )
        pipeline.unload_textual_inversion(
            tokens=["<s0>", "<s1>"], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2
        )
        ```
        """
    tokenizer = tokenizer or getattr(self, 'tokenizer', None)
    text_encoder = text_encoder or getattr(self, 'text_encoder', None)
    token_ids = []
    last_special_token_id = None
    if tokens:
        if isinstance(tokens, str):
            tokens = [tokens]
        for added_token_id, added_token in tokenizer.added_tokens_decoder.items(
            ):
            if not added_token.special:
                if added_token.content in tokens:
                    token_ids.append(added_token_id)
            else:
                last_special_token_id = added_token_id
        if len(token_ids) == 0:
            raise ValueError('No tokens to remove found')
    else:
        tokens = []
        for added_token_id, added_token in tokenizer.added_tokens_decoder.items(
            ):
            if not added_token.special:
                token_ids.append(added_token_id)
                tokens.append(added_token.content)
            else:
                last_special_token_id = added_token_id
    for token_id, token_to_remove in zip(token_ids, tokens):
        del tokenizer._added_tokens_decoder[token_id]
        del tokenizer._added_tokens_encoder[token_to_remove]
    key_id = 1
    for token_id in tokenizer.added_tokens_decoder:
        if (token_id > last_special_token_id and token_id > 
            last_special_token_id + key_id):
            token = tokenizer._added_tokens_decoder[token_id]
            tokenizer._added_tokens_decoder[last_special_token_id + key_id
                ] = token
            del tokenizer._added_tokens_decoder[token_id]
            tokenizer._added_tokens_encoder[token.content
                ] = last_special_token_id + key_id
            key_id += 1
    tokenizer._update_trie()
    text_embedding_dim = text_encoder.get_input_embeddings().embedding_dim
    temp_text_embedding_weights = text_encoder.get_input_embeddings().weight
    text_embedding_weights = temp_text_embedding_weights[:
        last_special_token_id + 1]
    to_append = []
    for i in range(last_special_token_id + 1, temp_text_embedding_weights.
        shape[0]):
        if i not in token_ids:
            to_append.append(temp_text_embedding_weights[i].unsqueeze(0))
    if len(to_append) > 0:
        to_append = torch.cat(to_append, dim=0)
        text_embedding_weights = torch.cat([text_embedding_weights,
            to_append], dim=0)
    text_embeddings_filtered = nn.Embedding(text_embedding_weights.shape[0],
        text_embedding_dim)
    text_embeddings_filtered.weight.data = text_embedding_weights
    text_encoder.set_input_embeddings(text_embeddings_filtered)
