def aggregate_attention(self, attention_maps, prompts, res: Union[int,
    Tuple[int]], from_where: List[str], is_cross: bool, select: int):
    out = [[] for x in range(self.batch_size)]
    if isinstance(res, int):
        num_pixels = res ** 2
        resolution = res, res
    else:
        num_pixels = res[0] * res[1]
        resolution = res[:2]
    for location in from_where:
        for bs_item in attention_maps[
            f"{location}_{'cross' if is_cross else 'self'}"]:
            for batch, item in enumerate(bs_item):
                if item.shape[1] == num_pixels:
                    cross_maps = item.reshape(len(prompts), -1, *resolution,
                        item.shape[-1])[select]
                    out[batch].append(cross_maps)
    out = torch.stack([torch.cat(x, dim=0) for x in out])
    out = out.sum(1) / out.shape[1]
    return out
