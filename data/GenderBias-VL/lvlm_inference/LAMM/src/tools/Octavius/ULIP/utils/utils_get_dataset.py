def get_dataset(train_transform, tokenizer, args, dataset_name=None):
    dataset_3d = Dataset_3D(args, tokenizer, dataset_name, train_transform)
    return dataset_3d.dataset
