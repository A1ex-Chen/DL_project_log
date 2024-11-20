def convert_attn_keys(key):
    """Convert the attention key from tuple format to the torch state format"""
    if key[0] == 'mid':
        assert key[1
            ] == 0, f'mid block only has one block but the index is {key[1]}'
        return (
            f'{key[0]}_block.attentions.{key[2]}.transformer_blocks.{key[3]}.attn2.processor'
            )
    return (
        f'{key[0]}_blocks.{key[1]}.attentions.{key[2]}.transformer_blocks.{key[3]}.attn2.processor'
        )
