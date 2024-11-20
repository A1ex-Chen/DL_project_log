def convert_tf2_checkpoint_to_pytorch(tf_checkpoint_path, config_path,
    pytorch_dump_path):
    logger.info(f'Loading model based on config from {config_path}...')
    config = BertConfig.from_json_file(config_path)
    model = BertModel(config)
    logger.info(f'Loading weights from checkpoint {tf_checkpoint_path}...')
    load_tf2_weights_in_bert(model, tf_checkpoint_path, config)
    logger.info(f'Saving PyTorch model to {pytorch_dump_path}...')
    torch.save(model.state_dict(), pytorch_dump_path)
