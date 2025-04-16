def build_retriever(train_dataset, test_dataset, retriever_type, **kwargs):
    build_fuc = retriever_dict[retriever_type]
    return build_fuc(train_dataset, test_dataset, **kwargs)
