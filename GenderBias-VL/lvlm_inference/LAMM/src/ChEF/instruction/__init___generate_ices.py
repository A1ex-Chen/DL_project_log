def generate_ices(self, prompts, batch_idx, batch_size):
    ice_idx = self.ice_idx_list[batch_idx * batch_size:(batch_idx + 1) *
        batch_size]
    ices = self.retriever.genetate_ice(ice_idx, prompts)
    return ices
