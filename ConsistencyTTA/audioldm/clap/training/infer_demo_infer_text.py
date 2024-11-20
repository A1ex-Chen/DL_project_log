def infer_text():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny'
    tmodel = 'roberta'
    enable_fusion = False
    fusion_type = 'aff_2d'
    pretrained = PRETRAINED_PATH
    model, model_cfg = create_model(amodel, tmodel, pretrained, precision=
        precision, device=device, enable_fusion=enable_fusion, fusion_type=
        fusion_type)
    text_data = ['I love the contrastive learning', 'I love the pretrain model'
        ]
    text_data = tokenizer(text_data)
    text_embed = model.get_text_embedding(text_data)
    print(text_embed.size())
