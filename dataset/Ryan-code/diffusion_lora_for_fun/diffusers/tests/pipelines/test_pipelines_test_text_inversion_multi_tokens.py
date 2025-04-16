def test_text_inversion_multi_tokens(self):
    pipe1 = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe1 = pipe1.to(torch_device)
    token1, token2 = '<*>', '<**>'
    ten1 = torch.ones((32,))
    ten2 = torch.ones((32,)) * 2
    num_tokens = len(pipe1.tokenizer)
    pipe1.load_textual_inversion(ten1, token=token1)
    pipe1.load_textual_inversion(ten2, token=token2)
    emb1 = pipe1.text_encoder.get_input_embeddings().weight
    pipe2 = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe2 = pipe2.to(torch_device)
    pipe2.load_textual_inversion([ten1, ten2], token=[token1, token2])
    emb2 = pipe2.text_encoder.get_input_embeddings().weight
    pipe3 = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe3 = pipe3.to(torch_device)
    pipe3.load_textual_inversion(torch.stack([ten1, ten2], dim=0), token=[
        token1, token2])
    emb3 = pipe3.text_encoder.get_input_embeddings().weight
    assert len(pipe1.tokenizer) == len(pipe2.tokenizer) == len(pipe3.tokenizer
        ) == num_tokens + 2
    assert pipe1.tokenizer.convert_tokens_to_ids(token1
        ) == pipe2.tokenizer.convert_tokens_to_ids(token1
        ) == pipe3.tokenizer.convert_tokens_to_ids(token1) == num_tokens
    assert pipe1.tokenizer.convert_tokens_to_ids(token2
        ) == pipe2.tokenizer.convert_tokens_to_ids(token2
        ) == pipe3.tokenizer.convert_tokens_to_ids(token2) == num_tokens + 1
    assert emb1[num_tokens].sum().item() == emb2[num_tokens].sum().item(
        ) == emb3[num_tokens].sum().item()
    assert emb1[num_tokens + 1].sum().item() == emb2[num_tokens + 1].sum(
        ).item() == emb3[num_tokens + 1].sum().item()
