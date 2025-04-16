@property
def do_perturbed_attention_guidance(self):
    return self._pag_scale > 0
