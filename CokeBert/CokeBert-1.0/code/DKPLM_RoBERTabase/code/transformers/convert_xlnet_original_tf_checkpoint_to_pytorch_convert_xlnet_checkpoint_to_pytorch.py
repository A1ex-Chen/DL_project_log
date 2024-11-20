def convert_xlnet_checkpoint_to_pytorch(tf_checkpoint_path,
    bert_config_file, pytorch_dump_folder_path, finetuning_task=None):
    config = XLNetConfig.from_json_file(bert_config_file)
    finetuning_task = finetuning_task.lower(
        ) if finetuning_task is not None else ''
    if finetuning_task in GLUE_TASKS_NUM_LABELS:
        print(
            'Building PyTorch XLNetForSequenceClassification model from configuration: {}'
            .format(str(config)))
        config.finetuning_task = finetuning_task
        config.num_labels = GLUE_TASKS_NUM_LABELS[finetuning_task]
        model = XLNetForSequenceClassification(config)
    elif 'squad' in finetuning_task:
        config.finetuning_task = finetuning_task
        model = XLNetForQuestionAnswering(config)
    else:
        model = XLNetLMHeadModel(config)
    load_tf_weights_in_xlnet(model, config, tf_checkpoint_path)
    pytorch_weights_dump_path = os.path.join(pytorch_dump_folder_path,
        WEIGHTS_NAME)
    pytorch_config_dump_path = os.path.join(pytorch_dump_folder_path,
        CONFIG_NAME)
    print('Save PyTorch model to {}'.format(os.path.abspath(
        pytorch_weights_dump_path)))
    torch.save(model.state_dict(), pytorch_weights_dump_path)
    print('Save configuration file to {}'.format(os.path.abspath(
        pytorch_config_dump_path)))
    with open(pytorch_config_dump_path, 'w', encoding='utf-8') as f:
        f.write(config.to_json_string())
