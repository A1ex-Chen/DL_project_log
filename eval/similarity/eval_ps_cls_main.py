def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_acc_dict = {k: (0.0) for k in MODEL_PATH.keys()}
    results = {}
    for model_name, model_path in MODEL_PATH.items():
        results[model_name] = {}
        phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.
            full_run_mode, args.task, 'train', model_path, device,
            contextual=args.contextual)
        phrase1_tensor.to(device)
        phrase2_tensor.to(device)
        label_tensor.to(device)
        train_dataset = ParaphraseDataset(phrase1_tensor, phrase2_tensor,
            label_tensor)
        phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.
            full_run_mode, args.task, 'validation', model_path, device,
            contextual=args.contextual)
        phrase1_tensor.to(device)
        phrase2_tensor.to(device)
        label_tensor.to(device)
        valid_dataset = ParaphraseDataset(phrase1_tensor, phrase2_tensor,
            label_tensor)
        phrase1_tensor, phrase2_tensor, label_tensor = get_data_emb(args.
            full_run_mode, args.task, 'test', model_path, device, shuffle=
            False, contextual=args.contextual)
        phrase1_tensor.to(device)
        phrase2_tensor.to(device)
        label_tensor.to(device)
        test_dataset = ParaphraseDataset(phrase1_tensor, phrase2_tensor,
            label_tensor)
        early_stop_callback = EarlyStopping(monitor='epoch_val_accuracy',
            min_delta=0.0, patience=10, verbose=True, mode='max')
        model = ProbingModel(input_dim=phrase1_tensor.shape[1] * 2,
            train_dataset=train_dataset, valid_dataset=valid_dataset,
            test_dataset=test_dataset).to(device)
        trainer = Trainer(max_epochs=100, min_epochs=3, auto_lr_find=False,
            auto_scale_batch_size=False, progress_bar_refresh_rate=10,
            callbacks=[early_stop_callback])
        trainer.fit(model)
        result = trainer.test(dataloaders=model.test_dataloader())
        print(result)
        result_dir = args.result_dir + ('contextual/' if args.contextual else
            'non_contextual/')
        if not exists(result_dir):
            makedirs(result_dir)
        output_fname = os.path.join(result_dir,
            f'{args.task}_{model_name}.json')
        with open(output_fname, 'w') as outfile:
            json.dump(result, outfile, indent=4)
        model_name_acc_dict[model_name] = result[0]['epoch_test_accuracy']
        print(f'\n finished {model_name}\n')
    for k, v in model_name_acc_dict.items():
        print(f' model: {k}\ttesting accuracy:\t{v:.4f}')
    print('Done with main --- {}'.format(args.task))
