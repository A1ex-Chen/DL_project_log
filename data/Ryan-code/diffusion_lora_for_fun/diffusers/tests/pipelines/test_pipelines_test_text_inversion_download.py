def test_text_inversion_download(self):
    pipe = StableDiffusionPipeline.from_pretrained(
        'hf-internal-testing/tiny-stable-diffusion-torch', safety_checker=None)
    pipe = pipe.to(torch_device)
    num_tokens = len(pipe.tokenizer)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ten = {'<*>': torch.ones((32,))}
        torch.save(ten, os.path.join(tmpdirname, 'learned_embeds.bin'))
        pipe.load_textual_inversion(tmpdirname)
        token = pipe.tokenizer.convert_tokens_to_ids('<*>')
        assert token == num_tokens, 'Added token must be at spot `num_tokens`'
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
            ) == 32
        assert pipe._maybe_convert_prompt('<*>', pipe.tokenizer) == '<*>'
        prompt = 'hey <*>'
        out = pipe(prompt, num_inference_steps=1, output_type='np').images
        assert out.shape == (1, 128, 128, 3)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ten = {'<**>': 2 * torch.ones((1, 32))}
        torch.save(ten, os.path.join(tmpdirname, 'learned_embeds.bin'))
        pipe.load_textual_inversion(tmpdirname, weight_name=
            'learned_embeds.bin')
        token = pipe.tokenizer.convert_tokens_to_ids('<**>')
        assert token == num_tokens + 1, 'Added token must be at spot `num_tokens`'
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
            ) == 64
        assert pipe._maybe_convert_prompt('<**>', pipe.tokenizer) == '<**>'
        prompt = 'hey <**>'
        out = pipe(prompt, num_inference_steps=1, output_type='np').images
        assert out.shape == (1, 128, 128, 3)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ten = {'<***>': torch.cat([3 * torch.ones((1, 32)), 4 * torch.ones(
            (1, 32)), 5 * torch.ones((1, 32))])}
        torch.save(ten, os.path.join(tmpdirname, 'learned_embeds.bin'))
        pipe.load_textual_inversion(tmpdirname)
        token = pipe.tokenizer.convert_tokens_to_ids('<***>')
        token_1 = pipe.tokenizer.convert_tokens_to_ids('<***>_1')
        token_2 = pipe.tokenizer.convert_tokens_to_ids('<***>_2')
        assert token == num_tokens + 2, 'Added token must be at spot `num_tokens`'
        assert token_1 == num_tokens + 3, 'Added token must be at spot `num_tokens`'
        assert token_2 == num_tokens + 4, 'Added token must be at spot `num_tokens`'
        assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item(
            ) == 96
        assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item(
            ) == 128
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
            ) == 160
        assert pipe._maybe_convert_prompt('<***>', pipe.tokenizer
            ) == '<***> <***>_1 <***>_2'
        prompt = 'hey <***>'
        out = pipe(prompt, num_inference_steps=1, output_type='np').images
        assert out.shape == (1, 128, 128, 3)
    with tempfile.TemporaryDirectory() as tmpdirname:
        ten = {'string_to_param': {'*': torch.cat([3 * torch.ones((1, 32)),
            4 * torch.ones((1, 32)), 5 * torch.ones((1, 32))])}, 'name':
            '<****>'}
        torch.save(ten, os.path.join(tmpdirname, 'a1111.bin'))
        pipe.load_textual_inversion(tmpdirname, weight_name='a1111.bin')
        token = pipe.tokenizer.convert_tokens_to_ids('<****>')
        token_1 = pipe.tokenizer.convert_tokens_to_ids('<****>_1')
        token_2 = pipe.tokenizer.convert_tokens_to_ids('<****>_2')
        assert token == num_tokens + 5, 'Added token must be at spot `num_tokens`'
        assert token_1 == num_tokens + 6, 'Added token must be at spot `num_tokens`'
        assert token_2 == num_tokens + 7, 'Added token must be at spot `num_tokens`'
        assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item(
            ) == 96
        assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item(
            ) == 128
        assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
            ) == 160
        assert pipe._maybe_convert_prompt('<****>', pipe.tokenizer
            ) == '<****> <****>_1 <****>_2'
        prompt = 'hey <****>'
        out = pipe(prompt, num_inference_steps=1, output_type='np').images
        assert out.shape == (1, 128, 128, 3)
    with tempfile.TemporaryDirectory() as tmpdirname1:
        with tempfile.TemporaryDirectory() as tmpdirname2:
            ten = {'<*****>': torch.ones((32,))}
            torch.save(ten, os.path.join(tmpdirname1, 'learned_embeds.bin'))
            ten = {'<******>': 2 * torch.ones((1, 32))}
            torch.save(ten, os.path.join(tmpdirname2, 'learned_embeds.bin'))
            pipe.load_textual_inversion([tmpdirname1, tmpdirname2])
            token = pipe.tokenizer.convert_tokens_to_ids('<*****>')
            assert token == num_tokens + 8, 'Added token must be at spot `num_tokens`'
            assert pipe.text_encoder.get_input_embeddings().weight[-2].sum(
                ).item() == 32
            assert pipe._maybe_convert_prompt('<*****>', pipe.tokenizer
                ) == '<*****>'
            token = pipe.tokenizer.convert_tokens_to_ids('<******>')
            assert token == num_tokens + 9, 'Added token must be at spot `num_tokens`'
            assert pipe.text_encoder.get_input_embeddings().weight[-1].sum(
                ).item() == 64
            assert pipe._maybe_convert_prompt('<******>', pipe.tokenizer
                ) == '<******>'
            prompt = 'hey <*****> <******>'
            out = pipe(prompt, num_inference_steps=1, output_type='np').images
            assert out.shape == (1, 128, 128, 3)
    ten = {'<x>': torch.ones((32,))}
    pipe.load_textual_inversion(ten)
    token = pipe.tokenizer.convert_tokens_to_ids('<x>')
    assert token == num_tokens + 10, 'Added token must be at spot `num_tokens`'
    assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
        ) == 32
    assert pipe._maybe_convert_prompt('<x>', pipe.tokenizer) == '<x>'
    prompt = 'hey <x>'
    out = pipe(prompt, num_inference_steps=1, output_type='np').images
    assert out.shape == (1, 128, 128, 3)
    ten1 = {'<xxxxx>': torch.ones((32,))}
    ten2 = {'<xxxxxx>': 2 * torch.ones((1, 32))}
    pipe.load_textual_inversion([ten1, ten2])
    token = pipe.tokenizer.convert_tokens_to_ids('<xxxxx>')
    assert token == num_tokens + 11, 'Added token must be at spot `num_tokens`'
    assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item(
        ) == 32
    assert pipe._maybe_convert_prompt('<xxxxx>', pipe.tokenizer) == '<xxxxx>'
    token = pipe.tokenizer.convert_tokens_to_ids('<xxxxxx>')
    assert token == num_tokens + 12, 'Added token must be at spot `num_tokens`'
    assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
        ) == 64
    assert pipe._maybe_convert_prompt('<xxxxxx>', pipe.tokenizer) == '<xxxxxx>'
    prompt = 'hey <xxxxx> <xxxxxx>'
    out = pipe(prompt, num_inference_steps=1, output_type='np').images
    assert out.shape == (1, 128, 128, 3)
    ten = {'string_to_param': {'*': torch.cat([3 * torch.ones((1, 32)), 4 *
        torch.ones((1, 32)), 5 * torch.ones((1, 32))])}, 'name': '<xxxx>'}
    pipe.load_textual_inversion(ten)
    token = pipe.tokenizer.convert_tokens_to_ids('<xxxx>')
    token_1 = pipe.tokenizer.convert_tokens_to_ids('<xxxx>_1')
    token_2 = pipe.tokenizer.convert_tokens_to_ids('<xxxx>_2')
    assert token == num_tokens + 13, 'Added token must be at spot `num_tokens`'
    assert token_1 == num_tokens + 14, 'Added token must be at spot `num_tokens`'
    assert token_2 == num_tokens + 15, 'Added token must be at spot `num_tokens`'
    assert pipe.text_encoder.get_input_embeddings().weight[-3].sum().item(
        ) == 96
    assert pipe.text_encoder.get_input_embeddings().weight[-2].sum().item(
        ) == 128
    assert pipe.text_encoder.get_input_embeddings().weight[-1].sum().item(
        ) == 160
    assert pipe._maybe_convert_prompt('<xxxx>', pipe.tokenizer
        ) == '<xxxx> <xxxx>_1 <xxxx>_2'
    prompt = 'hey <xxxx>'
    out = pipe(prompt, num_inference_steps=1, output_type='np').images
    assert out.shape == (1, 128, 128, 3)
    ten = {'<cat>': torch.ones(3, 32)}
    pipe.load_textual_inversion(ten)
    assert pipe._maybe_convert_prompt('<cat> <cat>', pipe.tokenizer
        ) == '<cat> <cat>_1 <cat>_2 <cat> <cat>_1 <cat>_2'
    prompt = 'hey <cat> <cat>'
    out = pipe(prompt, num_inference_steps=1, output_type='np').images
    assert out.shape == (1, 128, 128, 3)
