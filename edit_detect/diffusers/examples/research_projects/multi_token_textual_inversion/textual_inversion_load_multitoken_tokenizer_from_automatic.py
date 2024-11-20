def load_multitoken_tokenizer_from_automatic(tokenizer, text_encoder,
    automatic_dict, placeholder_token):
    """
    Automatic1111's tokens have format
    {'string_to_token': {'*': 265}, 'string_to_param': {'*': tensor([[ 0.0833,  0.0030,  0.0057,  ..., -0.0264, -0.0616, -0.0529],
        [ 0.0058, -0.0190, -0.0584,  ..., -0.0025, -0.0945, -0.0490],
        [ 0.0916,  0.0025,  0.0365,  ..., -0.0685, -0.0124,  0.0728],
        [ 0.0812, -0.0199, -0.0100,  ..., -0.0581, -0.0780,  0.0254]],
       requires_grad=True)}, 'name': 'FloralMarble-400', 'step': 399, 'sd_checkpoint': '4bdfc29c', 'sd_checkpoint_name': 'SD2.1-768'}
    """
    learned_embeds_dict = {}
    learned_embeds_dict[placeholder_token] = automatic_dict['string_to_param'][
        '*']
    load_multitoken_tokenizer(tokenizer, text_encoder, learned_embeds_dict)
