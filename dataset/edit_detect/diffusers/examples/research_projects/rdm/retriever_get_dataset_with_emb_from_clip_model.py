def get_dataset_with_emb_from_clip_model(dataset, clip_model,
    feature_extractor, image_column='image', index_name='embeddings'):
    return dataset.map(lambda example: {index_name:
        map_img_to_model_feature(clip_model.get_image_features,
        feature_extractor, [example[image_column]], clip_model.device)})
