def transform_fn(data_item):
    """
            Quantization transform function.

            Extracts and preprocess input data from dataloader item for quantization.
            Parameters:
               data_item: Tuple with data item produced by DataLoader during iteration
            Returns:
                input_tensor: Input data for quantization
            """
    img = data_item[0].numpy()
    input_tensor = prepare_input_tensor(img)
    return input_tensor
