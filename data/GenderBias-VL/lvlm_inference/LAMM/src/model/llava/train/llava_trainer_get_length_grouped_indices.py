def get_length_grouped_indices(lengths, batch_size, world_size, generator=
    None, merge=True):
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i:i + megabatch_size].tolist() for i in range(0,
        len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True
        ) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for
        megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in
        batch]
