def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0,
    lora_r=0, lora_target_modules=['q_proj', 'v_proj'], **lora_kargs):
    logging.info('Loading LLAMA')
    llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path,
        use_fast=False)
    llama_tokenizer.pad_token = '$$'
    if low_resource:
        llama_model = LlamaForCausalLM.from_pretrained(llama_model_path,
            torch_dtype=torch.float16, load_in_8bit=True, device_map={'':
            low_res_device})
    else:
        llama_model = LlamaForCausalLM.from_pretrained(llama_model_path,
            torch_dtype=torch.float16)
    if lora_r > 0:
        llama_model = prepare_model_for_int8_training(llama_model)
        loraconfig = LoraConfig(r=lora_r, bias='none', task_type=
            'CAUSAL_LM', target_modules=lora_target_modules, **lora_kargs)
        llama_model = get_peft_model(llama_model, loraconfig)
        llama_model.print_trainable_parameters()
    else:
        for name, param in llama_model.named_parameters():
            param.requires_grad = False
    logging.info('Loading LLAMA Done')
    return llama_model, llama_tokenizer
