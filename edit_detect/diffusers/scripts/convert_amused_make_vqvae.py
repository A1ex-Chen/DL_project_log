def make_vqvae(old_vae):
    new_vae = VQModel(act_fn='silu', block_out_channels=[128, 256, 256, 512,
        768], down_block_types=['DownEncoderBlock2D', 'DownEncoderBlock2D',
        'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        in_channels=3, latent_channels=64, layers_per_block=2,
        norm_num_groups=32, num_vq_embeddings=8192, out_channels=3,
        sample_size=32, up_block_types=['UpDecoderBlock2D',
        'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D',
        'UpDecoderBlock2D'], mid_block_add_attention=False,
        lookup_from_codebook=True)
    new_vae.to(device)
    new_state_dict = {}
    old_state_dict = old_vae.state_dict()
    new_state_dict['encoder.conv_in.weight'] = old_state_dict.pop(
        'encoder.conv_in.weight')
    new_state_dict['encoder.conv_in.bias'] = old_state_dict.pop(
        'encoder.conv_in.bias')
    convert_vae_block_state_dict(old_state_dict, 'encoder.down.0',
        new_state_dict, 'encoder.down_blocks.0')
    convert_vae_block_state_dict(old_state_dict, 'encoder.down.1',
        new_state_dict, 'encoder.down_blocks.1')
    convert_vae_block_state_dict(old_state_dict, 'encoder.down.2',
        new_state_dict, 'encoder.down_blocks.2')
    convert_vae_block_state_dict(old_state_dict, 'encoder.down.3',
        new_state_dict, 'encoder.down_blocks.3')
    convert_vae_block_state_dict(old_state_dict, 'encoder.down.4',
        new_state_dict, 'encoder.down_blocks.4')
    new_state_dict['encoder.mid_block.resnets.0.norm1.weight'
        ] = old_state_dict.pop('encoder.mid.block_1.norm1.weight')
    new_state_dict['encoder.mid_block.resnets.0.norm1.bias'
        ] = old_state_dict.pop('encoder.mid.block_1.norm1.bias')
    new_state_dict['encoder.mid_block.resnets.0.conv1.weight'
        ] = old_state_dict.pop('encoder.mid.block_1.conv1.weight')
    new_state_dict['encoder.mid_block.resnets.0.conv1.bias'
        ] = old_state_dict.pop('encoder.mid.block_1.conv1.bias')
    new_state_dict['encoder.mid_block.resnets.0.norm2.weight'
        ] = old_state_dict.pop('encoder.mid.block_1.norm2.weight')
    new_state_dict['encoder.mid_block.resnets.0.norm2.bias'
        ] = old_state_dict.pop('encoder.mid.block_1.norm2.bias')
    new_state_dict['encoder.mid_block.resnets.0.conv2.weight'
        ] = old_state_dict.pop('encoder.mid.block_1.conv2.weight')
    new_state_dict['encoder.mid_block.resnets.0.conv2.bias'
        ] = old_state_dict.pop('encoder.mid.block_1.conv2.bias')
    new_state_dict['encoder.mid_block.resnets.1.norm1.weight'
        ] = old_state_dict.pop('encoder.mid.block_2.norm1.weight')
    new_state_dict['encoder.mid_block.resnets.1.norm1.bias'
        ] = old_state_dict.pop('encoder.mid.block_2.norm1.bias')
    new_state_dict['encoder.mid_block.resnets.1.conv1.weight'
        ] = old_state_dict.pop('encoder.mid.block_2.conv1.weight')
    new_state_dict['encoder.mid_block.resnets.1.conv1.bias'
        ] = old_state_dict.pop('encoder.mid.block_2.conv1.bias')
    new_state_dict['encoder.mid_block.resnets.1.norm2.weight'
        ] = old_state_dict.pop('encoder.mid.block_2.norm2.weight')
    new_state_dict['encoder.mid_block.resnets.1.norm2.bias'
        ] = old_state_dict.pop('encoder.mid.block_2.norm2.bias')
    new_state_dict['encoder.mid_block.resnets.1.conv2.weight'
        ] = old_state_dict.pop('encoder.mid.block_2.conv2.weight')
    new_state_dict['encoder.mid_block.resnets.1.conv2.bias'
        ] = old_state_dict.pop('encoder.mid.block_2.conv2.bias')
    new_state_dict['encoder.conv_norm_out.weight'] = old_state_dict.pop(
        'encoder.norm_out.weight')
    new_state_dict['encoder.conv_norm_out.bias'] = old_state_dict.pop(
        'encoder.norm_out.bias')
    new_state_dict['encoder.conv_out.weight'] = old_state_dict.pop(
        'encoder.conv_out.weight')
    new_state_dict['encoder.conv_out.bias'] = old_state_dict.pop(
        'encoder.conv_out.bias')
    new_state_dict['quant_conv.weight'] = old_state_dict.pop(
        'quant_conv.weight')
    new_state_dict['quant_conv.bias'] = old_state_dict.pop('quant_conv.bias')
    new_state_dict['quantize.embedding.weight'] = old_state_dict.pop(
        'quantize.embedding.weight')
    new_state_dict['post_quant_conv.weight'] = old_state_dict.pop(
        'post_quant_conv.weight')
    new_state_dict['post_quant_conv.bias'] = old_state_dict.pop(
        'post_quant_conv.bias')
    new_state_dict['decoder.conv_in.weight'] = old_state_dict.pop(
        'decoder.conv_in.weight')
    new_state_dict['decoder.conv_in.bias'] = old_state_dict.pop(
        'decoder.conv_in.bias')
    new_state_dict['decoder.mid_block.resnets.0.norm1.weight'
        ] = old_state_dict.pop('decoder.mid.block_1.norm1.weight')
    new_state_dict['decoder.mid_block.resnets.0.norm1.bias'
        ] = old_state_dict.pop('decoder.mid.block_1.norm1.bias')
    new_state_dict['decoder.mid_block.resnets.0.conv1.weight'
        ] = old_state_dict.pop('decoder.mid.block_1.conv1.weight')
    new_state_dict['decoder.mid_block.resnets.0.conv1.bias'
        ] = old_state_dict.pop('decoder.mid.block_1.conv1.bias')
    new_state_dict['decoder.mid_block.resnets.0.norm2.weight'
        ] = old_state_dict.pop('decoder.mid.block_1.norm2.weight')
    new_state_dict['decoder.mid_block.resnets.0.norm2.bias'
        ] = old_state_dict.pop('decoder.mid.block_1.norm2.bias')
    new_state_dict['decoder.mid_block.resnets.0.conv2.weight'
        ] = old_state_dict.pop('decoder.mid.block_1.conv2.weight')
    new_state_dict['decoder.mid_block.resnets.0.conv2.bias'
        ] = old_state_dict.pop('decoder.mid.block_1.conv2.bias')
    new_state_dict['decoder.mid_block.resnets.1.norm1.weight'
        ] = old_state_dict.pop('decoder.mid.block_2.norm1.weight')
    new_state_dict['decoder.mid_block.resnets.1.norm1.bias'
        ] = old_state_dict.pop('decoder.mid.block_2.norm1.bias')
    new_state_dict['decoder.mid_block.resnets.1.conv1.weight'
        ] = old_state_dict.pop('decoder.mid.block_2.conv1.weight')
    new_state_dict['decoder.mid_block.resnets.1.conv1.bias'
        ] = old_state_dict.pop('decoder.mid.block_2.conv1.bias')
    new_state_dict['decoder.mid_block.resnets.1.norm2.weight'
        ] = old_state_dict.pop('decoder.mid.block_2.norm2.weight')
    new_state_dict['decoder.mid_block.resnets.1.norm2.bias'
        ] = old_state_dict.pop('decoder.mid.block_2.norm2.bias')
    new_state_dict['decoder.mid_block.resnets.1.conv2.weight'
        ] = old_state_dict.pop('decoder.mid.block_2.conv2.weight')
    new_state_dict['decoder.mid_block.resnets.1.conv2.bias'
        ] = old_state_dict.pop('decoder.mid.block_2.conv2.bias')
    convert_vae_block_state_dict(old_state_dict, 'decoder.up.0',
        new_state_dict, 'decoder.up_blocks.4')
    convert_vae_block_state_dict(old_state_dict, 'decoder.up.1',
        new_state_dict, 'decoder.up_blocks.3')
    convert_vae_block_state_dict(old_state_dict, 'decoder.up.2',
        new_state_dict, 'decoder.up_blocks.2')
    convert_vae_block_state_dict(old_state_dict, 'decoder.up.3',
        new_state_dict, 'decoder.up_blocks.1')
    convert_vae_block_state_dict(old_state_dict, 'decoder.up.4',
        new_state_dict, 'decoder.up_blocks.0')
    new_state_dict['decoder.conv_norm_out.weight'] = old_state_dict.pop(
        'decoder.norm_out.weight')
    new_state_dict['decoder.conv_norm_out.bias'] = old_state_dict.pop(
        'decoder.norm_out.bias')
    new_state_dict['decoder.conv_out.weight'] = old_state_dict.pop(
        'decoder.conv_out.weight')
    new_state_dict['decoder.conv_out.bias'] = old_state_dict.pop(
        'decoder.conv_out.bias')
    assert len(old_state_dict.keys()) == 0
    new_vae.load_state_dict(new_state_dict)
    input = torch.randn((1, 3, 512, 512), device=device)
    input = input.clamp(-1, 1)
    old_encoder_output = old_vae.quant_conv(old_vae.encoder(input))
    new_encoder_output = new_vae.quant_conv(new_vae.encoder(input))
    assert (old_encoder_output == new_encoder_output).all()
    old_decoder_output = old_vae.decoder(old_vae.post_quant_conv(
        old_encoder_output))
    new_decoder_output = new_vae.decoder(new_vae.post_quant_conv(
        new_encoder_output))
    print('kipping vae decoder equivalence check')
    print(
        f'vae decoder diff {(old_decoder_output - new_decoder_output).float().abs().sum()}'
        )
    old_output = old_vae(input)[0]
    new_output = new_vae(input)[0]
    print('skipping full vae equivalence check')
    print(f'vae full diff {(old_output - new_output).float().abs().sum()}')
    return new_vae
