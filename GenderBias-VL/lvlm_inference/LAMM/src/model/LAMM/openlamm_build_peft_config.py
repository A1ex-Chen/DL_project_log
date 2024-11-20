def build_peft_config(self):
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=
        False, r=self.args['lora_r'], lora_alpha=self.args['lora_alpha'],
        lora_dropout=self.args['lora_dropout'], target_modules=self.args[
        'lora_target_modules'])
    return peft_config
