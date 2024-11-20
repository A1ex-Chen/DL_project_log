def update_ema(self):
    super().update_ema()
    do_ema_update(source_model=self.vae.decoder, shadow_models=[self.
        ema_vae_decoder], decay_consts=[self.ema_decay])
    do_ema_update(source_model=self.vae.post_quant_conv, shadow_models=[
        self.ema_vae_pqconv], decay_consts=[self.ema_decay])
