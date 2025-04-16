def layer_hook(module, inp, out):
    features_out_hook.append(out.data.cpu().numpy())
