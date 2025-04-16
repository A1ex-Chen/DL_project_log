def get_dataset_with_emb_from_model(dataset, model, feature_extractor,
    image_column='image', index_name='embeddings'):
    return dataset.map(lambda example: {index_name:
        map_img_to_model_feature(model, feature_extractor, [example[
        image_column]], model.device)})
