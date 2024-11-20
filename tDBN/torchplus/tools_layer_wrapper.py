def layer_wrapper(layer_class):


    class DefaultArgLayer(layer_class):

        def __init__(self, *args, **kw):
            pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
            kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
            for key, val in kwargs.items():
                if key not in kw and kw_to_pos[key] > len(args):
                    kw[key] = val
            super().__init__(*args, **kw)
    return DefaultArgLayer
