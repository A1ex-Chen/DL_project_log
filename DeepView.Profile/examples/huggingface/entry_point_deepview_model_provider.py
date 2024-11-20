def deepview_model_provider():
    return AutoModelForCausalLM.from_pretrained(model_id, is_decoder=True
        ).cuda()
