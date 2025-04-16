def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')
