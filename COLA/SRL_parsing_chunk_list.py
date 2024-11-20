def chunk_list(seq_list, batch_size):
    out = []
    for start_idx in range(0, len(seq_list), batch_size):
        end_idx = min(start_idx + batch_size, len(seq_list))
        out.append(seq_list[start_idx:end_idx])
    return out
