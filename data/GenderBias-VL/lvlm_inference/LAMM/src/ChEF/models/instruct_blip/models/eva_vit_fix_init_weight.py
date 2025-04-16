def fix_init_weight(self):

    def rescale(param, layer_id):
        param.div_(math.sqrt(2.0 * layer_id))
    for layer_id, layer in enumerate(self.blocks):
        rescale(layer.attn.proj.weight.data, layer_id + 1)
        rescale(layer.mlp.fc2.weight.data, layer_id + 1)
