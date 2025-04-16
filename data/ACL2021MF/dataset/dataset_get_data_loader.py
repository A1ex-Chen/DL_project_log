def get_data_loader(dataset, batch_size):
    collate_fn = lambda d: data_wrapper(d, dataset.tokenizer, dataset.
        decoder_start_token_id)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0,
        collate_fn=collate_fn)
