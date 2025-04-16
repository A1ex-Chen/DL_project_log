@torch.jit.export
def reorder_incremental_state(self, incremental_states: List[Dict[str, Dict
    [str, Optional[Tensor]]]], new_order):
    if not self.has_incremental_states():
        return
    for i, model in enumerate(self.models):
        if hasattr(model, 'gpt_model'):
            model.gpt_model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order)
        else:
            model.decoder.reorder_incremental_state_scripting(
                incremental_states[i], new_order)
