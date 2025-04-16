def gather_features(audio_features, text_features, audio_features_mlp=None,
    text_features_mlp=None, local_loss=False, gather_with_grad=False, rank=
    0, world_size=1, use_horovod=False, mlp_loss=False):
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_audio_features = hvd.allgather(audio_features)
            all_text_features = hvd.allgather(text_features)
            if mlp_loss:
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                all_text_features_mlp = hvd.allgather(text_features_mlp)
        else:
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)
                all_text_features = hvd.allgather(text_features)
                if mlp_loss:
                    all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                    all_text_features_mlp = hvd.allgather(text_features_mlp)
            if not local_loss:
                gathered_audio_features = list(all_audio_features.chunk(
                    world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(
                    world_size, dim=0))
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                all_audio_features = torch.cat(gathered_audio_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if mlp_loss:
                    gathered_audio_features_mlp = list(all_audio_features_mlp
                        .chunk(world_size, dim=0))
                    gathered_text_features_mlp = list(all_text_features_mlp
                        .chunk(world_size, dim=0))
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_text_features_mlp[rank] = text_features_mlp
                    all_audio_features_mlp = torch.cat(
                        gathered_audio_features_mlp, dim=0)
                    all_text_features_mlp = torch.cat(
                        gathered_text_features_mlp, dim=0)
    elif gather_with_grad:
        all_audio_features = torch.cat(torch.distributed.nn.all_gather(
            audio_features), dim=0)
        all_text_features = torch.cat(torch.distributed.nn.all_gather(
            text_features), dim=0)
        if mlp_loss:
            all_audio_features_mlp = torch.cat(torch.distributed.nn.
                all_gather(audio_features_mlp), dim=0)
            all_text_features_mlp = torch.cat(torch.distributed.nn.
                all_gather(text_features_mlp), dim=0)
    else:
        gathered_audio_features = [torch.zeros_like(audio_features) for _ in
            range(world_size)]
        gathered_text_features = [torch.zeros_like(text_features) for _ in
            range(world_size)]
        dist.all_gather(gathered_audio_features, audio_features)
        dist.all_gather(gathered_text_features, text_features)
        if mlp_loss:
            gathered_audio_features_mlp = [torch.zeros_like(
                audio_features_mlp) for _ in range(world_size)]
            gathered_text_features_mlp = [torch.zeros_like(
                text_features_mlp) for _ in range(world_size)]
            dist.all_gather(gathered_audio_features_mlp, audio_features_mlp)
            dist.all_gather(gathered_text_features_mlp, text_features_mlp)
        if not local_loss:
            gathered_audio_features[rank] = audio_features
            gathered_text_features[rank] = text_features
            if mlp_loss:
                gathered_audio_features_mlp[rank] = audio_features_mlp
                gathered_text_features_mlp[rank] = text_features_mlp
        all_audio_features = torch.cat(gathered_audio_features, dim=0)
        all_text_features = torch.cat(gathered_text_features, dim=0)
        if mlp_loss:
            all_audio_features_mlp = torch.cat(gathered_audio_features_mlp,
                dim=0)
            all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0
                )
    if mlp_loss:
        return (all_audio_features, all_text_features,
            all_audio_features_mlp, all_text_features_mlp)
    else:
        return all_audio_features, all_text_features
