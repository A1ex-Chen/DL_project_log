def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
    data_args) ->Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=
        data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None,
        data_collator=data_collator)
