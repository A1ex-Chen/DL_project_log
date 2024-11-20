def _get_eval_sampler(self, eval_dataset: Dataset) ->Optional[torch.utils.
    data.sampler.Sampler]:
    if is_torch_tpu_available():
        return SequentialDistributedSampler(eval_dataset, num_replicas=xm.
            xrt_world_size(), rank=xm.get_ordinal())
    elif self.args.local_rank != -1:
        return SequentialDistributedSampler(eval_dataset)
    else:
        return SequentialSampler(eval_dataset)
