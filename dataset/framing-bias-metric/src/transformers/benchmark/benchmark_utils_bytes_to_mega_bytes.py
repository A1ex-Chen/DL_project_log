def bytes_to_mega_bytes(memory_amount: int) ->int:
    """Utility to convert a number of bytes (int) into a number of mega bytes (int)"""
    return memory_amount >> 20
