def get_position_emb_index(distance, num_buckets=16, max_distance=128,
    right=False):
    max_exact = num_buckets // 2
    if distance < max_exact:
        return distance if not right else distance + num_buckets
    else:
        pos = max_exact + math.log(distance / max_exact) / math.log(
            max_distance / max_exact) * (num_buckets - max_exact)
        pos = int(min(pos, num_buckets - 1))
        return pos if not right else pos + num_buckets
