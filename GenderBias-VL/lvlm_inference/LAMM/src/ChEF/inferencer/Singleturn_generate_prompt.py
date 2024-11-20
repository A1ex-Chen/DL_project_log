def generate_prompt(self, model, batch, batch_idx):
    if self.CoT:
        return self.instruction_handler.generate_CoT_prompt(model, batch)
    return self.instruction_handler.generate_singleturn_prompt(batch, batch_idx
        ), None
