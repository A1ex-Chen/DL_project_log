def get_test_dataloader(test_file, text_column, audio_column, batch_size):
    extension = test_file.split('.')[-1]
    data_files = {'test': test_file}
    test_set = datasets.load_dataset(extension, data_files=data_files)['test']
    test_dataset = Text2AudioDataset(test_set, text_column=text_column,
        audio_column=audio_column, num_examples=-1, prefix='',
        target_length=TARGET_LENGTH, augment=False)
    try:
        logger.info(f'Num instances in test dataset: {len(test_dataset)}.')
    except:
        print(f'Num instances in test dataset: {len(test_dataset)}.')
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=
        batch_size, num_workers=4, collate_fn=test_dataset.collate_fn)
    return test_dataloader
