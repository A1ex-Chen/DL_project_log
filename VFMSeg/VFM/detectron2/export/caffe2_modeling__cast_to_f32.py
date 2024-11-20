def _cast_to_f32(f64):
    return struct.unpack('f', struct.pack('f', f64))[0]
