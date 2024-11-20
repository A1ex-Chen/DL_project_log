def check_if_lora_correctly_set(self, model) ->bool:
    """
        Checks if the LoRA layers are correctly set with peft
        """
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return True
    return False
