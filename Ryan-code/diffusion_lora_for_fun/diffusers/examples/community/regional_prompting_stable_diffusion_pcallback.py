def pcallback(s_self, step: int, timestep: int, latents: torch.Tensor,
    selfs=None):
    if 'PRO' in mode:
        self.step = step
        if len(self.attnmaps_sizes) > 3:
            self.history[step] = self.attnmaps.copy()
            for hw in self.attnmaps_sizes:
                allmasks = []
                basemasks = [None] * batch
                for tt, th in zip(target_tokens, thresholds):
                    for b in range(batch):
                        key = f'{tt}-{b}'
                        _, mask, _ = makepmask(self, self.attnmaps[key], hw
                            [0], hw[1], th, step)
                        mask = mask.unsqueeze(0).unsqueeze(-1)
                        if self.ex:
                            allmasks[b::batch] = [(x - mask) for x in
                                allmasks[b::batch]]
                            allmasks[b::batch] = [torch.where(x > 0, 1, 0) for
                                x in allmasks[b::batch]]
                        allmasks.append(mask)
                        basemasks[b] = mask if basemasks[b
                            ] is None else basemasks[b] + mask
                basemasks = [(1 - mask) for mask in basemasks]
                basemasks = [torch.where(x > 0, 1, 0) for x in basemasks]
                allmasks = basemasks + allmasks
                self.attnmasks[hw] = torch.cat(allmasks)
            self.maskready = True
    return latents
