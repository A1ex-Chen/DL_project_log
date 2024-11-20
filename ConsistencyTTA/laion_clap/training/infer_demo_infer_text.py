def infer_text():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    precision = 'fp32'
    amodel = 'HTSAT-tiny'
    tmodel = 'roberta'
    enable_fusion = True
    fusion_type = 'aff_2d'
    pretrained = '/home/la/kechen/Research/KE_CLAP/ckpt/fusion_best.pt'
    model, model_cfg = create_model(amodel, tmodel, pretrained, precision=
        precision, device=device, enable_fusion=enable_fusion, fusion_type=
        fusion_type)
    text_data = ['I love the contrastive learning', 'I love the pretrain model'
        ]
    text_data = tokenizer(text_data)
    model.eval()
    text_embed = model.get_text_embedding(text_data)
    text_embed = text_embed.detach().cpu().numpy()
    print(text_embed)
    print(text_embed.shape)
