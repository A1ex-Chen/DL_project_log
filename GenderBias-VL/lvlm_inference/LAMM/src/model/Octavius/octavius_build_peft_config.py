def build_peft_config(self):
    print(f'Build PEFT model with LoRA-MoE.')
    peft_config = MoeLoraConfig(task_type=TaskType.CAUSAL_LM,
        inference_mode=False, r=self.args['lora_r'], num_experts=self.args[
        'moe_lora_num_experts'], gate_mode=self.args['moe_gate_mode'],
        lora_alpha=self.args['lora_alpha'], lora_dropout=self.args[
        'lora_dropout'], target_modules=self.args['lora_target_modules'])
    return peft_config
