def get_dummy_components(self, time_cond_proj_dim=None):
    torch.manual_seed(0)
    unet = Kandinsky3UNet(in_channels=4, time_embedding_dim=4, groups=2,
        attention_head_dim=4, layers_per_block=3, block_out_channels=(32, 
        64), cross_attention_dim=4, encoder_hid_dim=32)
    scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012,
        steps_offset=1, beta_schedule='squaredcos_cap_v2', clip_sample=True,
        thresholding=False)
    torch.manual_seed(0)
    movq = self.dummy_movq
    torch.manual_seed(0)
    text_encoder = T5EncoderModel.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(
        'hf-internal-testing/tiny-random-t5')
    components = {'unet': unet, 'scheduler': scheduler, 'movq': movq,
        'text_encoder': text_encoder, 'tokenizer': tokenizer}
    return components
