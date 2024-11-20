def tmp_train():
    config: Config = Config()
    accelerator = Accelerator(log_with=['wandb', LoggerType.TENSORBOARD])
    g = torch.Generator(device=config.device).manual_seed(config.seed)
    pipe, unet, vae, scheduler = ModelSched.get_model_sched(model_type=
        config.model_type, model_id=config.model_id)
    loss_fn = LossFn.get_loss_fn(metric=config.loss_metric)
    pipe, unet, vae, scheduler, loss_fn = ModelSched.all_to_device(pipe,
        unet, vae, scheduler, loss_fn, device=config.device)
    pipe.unet, unet, vae = ModelSched.all_compile(pipe.unet, unet, vae)
    image_size: Union[int, List[int], Tuple[int]
        ] = pipe.unet.config.sample_size
    ds = get_dataset(root=config.ds_root, size=image_size, repeat=config.
        repeat, vmin_out=config.vmin_out, vmax_out=config.vmax_out)
    dl = get_dataloader(dataset=ds, batch_size=config.batch_size)
    accelerator.init_trackers(project_name=config.project, config=config,
        init_kwargs={'wandb': {'name': config.name, 'id': config.name,
        'settings': wandb.Settings(start_method='fork')}})
    os.makedirs(config.name, exist_ok=True)
    with open(os.path.join(config.name, 'config.json'), 'w') as f:
        json.dump(Dataclass2Dict(data=config), f, indent=4)
    try:
        for batch_idx, batch in enumerate(dl):
            if batch_idx >= 500:
                break
            img, noise, img_id = batch
            img, noise = ModelSched.all_to_device(img, noise, device=config
                .device)
            noise = Optimizer.get_tainable_param(noise)
            optim = Optimizer.optim_generator(name=config.optim_type, lr=
                config.lr)([noise])
            print(f'Batch {batch_idx}')
            progress_bar = tqdm(total=config.max_iter)
            progress_bar.set_description(f'Batch {batch_idx}')
            loss_log = []
            loss_vec_log = []
            step_log = []
            sample_log = []
            for step in tqdm(range(config.max_iter)):
                pred = single_step_denoise(pipeline=pipe, latents=noise.
                    float(), from_t=0).float()
                loss_vec = loss_fn(img.float(), pred)
                loss = loss_vec.mean()
                optim.zero_grad()
                loss.backward()
                optim.step()
                progress_bar.update(1)
                logs = {'AvgLoss': loss.detach().item()}
                progress_bar.set_postfix(**logs)
                loss_log.append(loss.detach().cpu().item())
                loss_vec_log.append(loss_vec.detach().cpu())
                step_log.append(step)
                sample_log.append(ds.tensor2imgs(auto_make_grid(pred[:
                    config.grid_size]), cnvt_range=False, is_detach=True,
                    to_device='cpu'))
                torch.cuda.empty_cache()
            fig, ax = plt.subplots()
            ax.plot(step_log, loss_log)
            wandb_video = wandb.Video(np.stack(ds.tensor2imgs(sample_log,
                cnvt_range=True, to_unint8=True, to_np=True)), fps=config.
                max_iter // config.animate_duration)
            pil_ls = ds.tensor2imgs(sample_log, to_pil=True)
            accelerator.log({'loss': fig, 'sample_evol': wandb_video,
                'final_loss': loss_log[-1], 'final_sample': wandb.Image(
                pil_ls[-1])}, step=batch_idx)
            work_dir = os.path.join(config.name, f'batch{batch_idx}')
            os.makedirs(work_dir, exist_ok=True)
            os.makedirs(os.path.join(work_dir, config.
                training_sample_folder), exist_ok=True)
            batch_config = Dataclass2Dict(data=config)
            batch_config['batch_idx'] = batch_idx
            with open(os.path.join(work_dir, 'config.json'), 'w') as f:
                json.dump(batch_config, f, indent=4)
            for i, sample in enumerate(pil_ls):
                sample.save(os.path.join(work_dir, config.
                    training_sample_folder, f'sample{i}.jpg'))
            pil_ls[-1].save(os.path.join(work_dir, f'final.jpg'))
            pil_ls[0].save(os.path.join(work_dir, f'animate.gif'), save_all
                =True, append_images=pil_ls[1:], duration=config.
                animate_duration, loop=0)
            save_file({'loss_vec_log': torch.stack(loss_vec_log),
                'final_sample': sample_log[-1]}, os.path.join(work_dir,
                f'record.safetensors'))
            fig.savefig(os.path.join(work_dir, f'loss.jpg'))
            torch.cuda.empty_cache()
    finally:
        accelerator.end_training()
