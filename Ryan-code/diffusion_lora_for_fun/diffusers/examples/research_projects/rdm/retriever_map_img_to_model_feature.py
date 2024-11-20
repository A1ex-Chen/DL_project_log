def map_img_to_model_feature(model, feature_extractor, imgs, device):
    for i, image in enumerate(imgs):
        if not image.mode == 'RGB':
            imgs[i] = image.convert('RGB')
    imgs = normalize_images(imgs)
    retrieved_images = preprocess_images(imgs, feature_extractor).to(device)
    image_embeddings = model(retrieved_images)
    image_embeddings = image_embeddings / torch.linalg.norm(image_embeddings,
        dim=-1, keepdim=True)
    image_embeddings = image_embeddings[None, ...]
    return image_embeddings.cpu().detach().numpy()[0][0]
