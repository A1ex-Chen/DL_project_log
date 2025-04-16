def get_first_cross_attention(block):
    return block.attentions[0].transformer_blocks[0].attn2
