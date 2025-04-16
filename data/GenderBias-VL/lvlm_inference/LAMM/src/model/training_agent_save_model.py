def save_model(self, path, current_step):
    trainable_params = [k for k, v in self.ds_engine.module.
        named_parameters() if v.requires_grad]
    state_dict = None
    if self.ds_engine.zero_optimization_partition_weights():
        if self.ds_engine.zero_gather_16bit_weights_on_model_save():
            state_dict = self.ds_engine._zero3_consolidated_16bit_state_dict()
        else:
            raise NotImplementedError
    else:
        state_dict = self.ds_engine.module.state_dict()
    if deepspeed.comm.get_rank() == 0:
        checkpoint = OrderedDict((k, state_dict[k]) for k in trainable_params)
        if current_step <= 0:
            torch.save(checkpoint, f'{path}/pytorch_model.pt')
        else:
            torch.save(checkpoint, f'{path}/pytorch_model_ep{current_step}.pt')
        self.model.llama_tokenizer.save_pretrained(path)
        self.model.llama_model.config.save_pretrained(path)
        print(f'[!] save model into {path}')
