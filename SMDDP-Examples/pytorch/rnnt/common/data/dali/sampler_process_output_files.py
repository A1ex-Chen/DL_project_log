def process_output_files(self, output_files):
    print_once('Launching vectorized bucketing sampler')
    names = list(output_files)
    lengths = [output_files[name]['duration'] for name in names]
    labels = np.array([output_files[name]['label'] for name in names])
    dur = torch.tensor(lengths, device='cuda')
    len_ids = dur.argsort()
    buckets = len_ids.tensor_split(self.num_buckets)
    padded_buckets = torch.nn.utils.rnn.pad_sequence(buckets, padding_value
        =-1, batch_first=True)
    with torch.random.fork_rng(devices=range(torch.cuda.device_count())):
        torch.random.manual_seed(self.seed)
        self.seed += 1
        buckets_shuffler = torch.rand(self.num_epochs, *padded_buckets.
            shape, device='cuda')
        shuffle_columnvise = buckets_shuffler.argsort(dim=2)
        epochs, num_buckets, samples = shuffle_columnvise.shape
        shift = torch.arange(0, samples * num_buckets, samples, device='cuda'
            ).view(1, -1, 1)
        shuffle_globalvise = shuffle_columnvise + shift
        shuffled_buckets = padded_buckets.take(shuffle_globalvise)
        gbs = self.batch_size * self.num_workers
        unpadded = shuffled_buckets[shuffled_buckets != -1].view(epochs, -1)
        epochs, samples = unpadded.shape
        to_drop = samples - samples // gbs * gbs
        mask = torch.ones_like(unpadded, dtype=bool, device='cuda')
        removed_samples = torch.rand(unpadded.shape, device='cuda').argsort(dim
            =1)[:, :to_drop]
        epoch_idx = torch.arange(self.num_epochs).view(-1, 1).expand(self.
            num_epochs, to_drop)
        mask[epoch_idx.flatten(), removed_samples.flatten()] = False
        batch_aligned = unpadded[mask].view(self.num_epochs, -1, self.
            batch_size)
        _, num_iterations, _ = batch_aligned.shape
        epochs, num_batches, bs = batch_aligned.view(self.num_epochs, -1, gbs
            ).shape
        new_order = torch.rand(epochs, num_batches, device='cuda')
        nwo = new_order.argsort(dim=1).view(-1, num_batches, 1
            ) * bs + torch.arange(0, bs, 1, device='cuda').view(1, 1, -1
            ) + torch.arange(0, epochs * num_batches * bs, num_batches * bs,
            device='cuda').view(-1, 1, 1)
        out = batch_aligned.take(nwo)
        if self.pre_sort:
            pert_range = self.config_data['speed_perturbation']['max_rate'
                ] - self.config_data['speed_perturbation']['min_rate']
            self.pert_coeff = torch.rand(out.size(0), out.size(1), out.size
                (2), device='cuda') * pert_range + self.config_data[
                'speed_perturbation']['min_rate']
            dur_after_pert = dur[out] * self.pert_coeff
            idx_asc = dur_after_pert.argsort(dim=-1)
            idx_des = torch.flip(idx_asc, dims=[-1])
            idx_mix = torch.ones_like(idx_asc)
            idx_mix[:, :, ::2] = idx_asc[:, :, :idx_asc.size(-1) // 2]
            idx_mix[:, :, 1::2] = idx_des[:, :, :idx_des.size(-1) // 2]
            out = torch.gather(out, 2, idx_mix)
            self.pert_coeff = torch.gather(self.pert_coeff, 2, idx_mix)
    if self.dist_sampler:
        out = out.view(epochs, -1, self.num_workers, self.batch_size).moveaxis(
            2, 0)
        out = out[self.rank]
        if self.pre_sort:
            self.pert_coeff = self.pert_coeff.view(epochs, -1, self.
                num_workers, self.batch_size).moveaxis(2, 0)
            self.pert_coeff = self.pert_coeff[self.rank].cpu()
    self.dataset_size = num_iterations * self.batch_size
    out = out.cpu()
    return np.array(names)[out.flatten()].tolist(), np.array(labels)[out.
        flatten()].tolist()
