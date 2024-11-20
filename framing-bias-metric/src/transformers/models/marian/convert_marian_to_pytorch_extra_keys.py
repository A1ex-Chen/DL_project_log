@property
def extra_keys(self):
    extra = []
    for k in self.state_keys:
        if k.startswith('encoder_l') or k.startswith('decoder_l') or k in [
            CONFIG_KEY, 'Wemb', 'Wpos', 'decoder_ff_logit_out_b']:
            continue
        else:
            extra.append(k)
    return extra
