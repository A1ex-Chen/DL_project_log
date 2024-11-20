@staticmethod
def _reorder_cache(past, beam_idx):
    return tuple(tf.gather(layer_past, beam_idx, axis=1) for layer_past in past
        )
