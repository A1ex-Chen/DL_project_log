def initialize_language_model(self, llm_ckpt_path):
    self.llama_model = LlamaForCausalLM.from_pretrained(llm_ckpt_path)
    if self.stage == 1:
        print('Freeze language decoder for stage 1 trainning')
        self.llama_model.model.requires_grad_(False)
    if self.stage == 2:
        self.gradient_checkpointing = self.args['gradient_checkpointing']
        if self.gradient_checkpointing:
            print('Enable gradient checkpointing for SFT')
            self.llama_model.model.gradient_checkpointing = True
        print('Enable language decoder for stage 2 training')
        self.llama_model.model.requires_grad_(True)
