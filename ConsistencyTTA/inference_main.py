def main():
    args = parse_args()
    seed_all(args.seed)
    sr = 16000
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps' if
        torch.backends.mps.is_available() else 'cpu')
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    if 'hf_model' not in train_args:
        train_args['hf_model'] = None
    name = 'audioldm-s-full'
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.to(device), stft.to(device)
    assert args.stage == train_args.stage, 'Stage mismatch between training and eval.'
    if args.stage == 1:
        model_class = AudioGDM
    elif train_args.finetune_vae:
        model_class = AudioLCM_FTVAE
    else:
        model_class = AudioLCM
    model = model_class(text_encoder_name=train_args.text_encoder_name,
        scheduler_name=train_args.scheduler_name, unet_model_name=
        train_args.unet_model_name, unet_model_config_path=train_args.
        unet_model_config, snr_gamma=train_args.snr_gamma,
        freeze_text_encoder=train_args.freeze_text_encoder, uncondition=
        train_args.uncondition, use_edm=train_args.use_edm, use_karras=
        train_args.use_karras, use_lora=train_args.use_lora,
        target_ema_decay=train_args.target_ema_decay, ema_decay=train_args.
        ema_decay, num_diffusion_steps=train_args.num_diffusion_steps,
        teacher_guidance_scale=train_args.teacher_guidance_scale, loss_type
        =train_args.loss_type, vae=vae).to(device)
    print('Loading ConsistencyTTA model weights...')
    model.load_pretrained(torch.load(args.model, map_location='cpu'))
    model.eval()
    if hasattr(model, 'vae') and model.vae is not None:
        print("Replacing VAE with model's VAE.")
        vae = model.vae
    sched_class = HeunDiscreteScheduler if args.use_edm else DDIMScheduler
    scheduler = sched_class.from_pretrained(train_args.scheduler_name,
        subfolder='scheduler')
    if args.use_karras:
        if args.use_edm:
            print('Using Karras noise schedule.')
            scheduler.use_karras_sigmas = True
        else:
            ValueError(
                'Karras noise schedule can only be used with the Heun scheduler.'
                )
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ''
    text_prompts = [json.loads(line)[args.text_key] for line in open(args.
        test_file).readlines()]
    text_prompts = [(prefix + inp) for inp in text_prompts]
    if args.num_test_instances != -1:
        text_prompts = text_prompts[:args.num_test_instances]
    num_steps, batch_size = args.num_steps, args.batch_size
    guidance_input, guidance_post = (args.guidance_scale_input, args.
        guidance_scale_post)
    all_outputs, all_mels = [], []
    for k in tqdm(range(0, len(text_prompts), batch_size)):
        text = text_prompts[k:k + batch_size]
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16,
                enabled=args.use_bf16):
                latents = model.inference(text, scheduler, num_steps=
                    num_steps, num_samples=1, guidance_scale_input=
                    guidance_input, guidance_scale_post=guidance_post,
                    use_edm=args.use_edm, use_ema=args.use_ema,
                    query_teacher=args.query_teacher)
                use_ema_decoder = args.use_ema and hasattr(vae, 'ema_decoder'
                    ) and vae.ema_decoder is not None
                mel = vae.decode_first_stage(latents.float(), use_ema=
                    use_ema_decoder)
                wave = vae.decode_to_waveform(mel.float())
            all_outputs += [w[:int(sr * 10)] for w in wave]
            all_mels += [mel.detach().cpu().float()]
    exp_id = str(int(time.time()))
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    output_dir = (
        f"outputs/{exp_id}_{'_'.join(args.model.split('/')[1:-1])}_steps_{num_steps}_guidance_{guidance_input}"
        )
    os.makedirs(output_dir, exist_ok=True)
    for j, wav in enumerate(all_outputs):
        sf.write(f'{output_dir}/output_{j}.wav', wav, samplerate=sr)
    torch.save(torch.cat(all_mels), f'{output_dir}/all_mels.pt')
    wandb.init(project='Text to Audio Generation Evaluation')
    evaluator = EvaluationHelper(sampling_rate=sr, device='cuda:0')
    result = evaluator.main(dataset_json_path=args.test_file,
        groundtruth_path=args.test_references, generated_files_path=
        output_dir, mel_path=f'{output_dir}/all_mels.pt')
    result['Steps'] = num_steps
    result['Guidance Scale'] = guidance_input
    result['Test Instances'] = len(text_prompts)
    wandb.log(result)
    result['scheduler_config'] = dict(scheduler.config)
    result['args'] = dict(vars(args))
    result['output_dir'] = output_dir
    with open('outputs/summary.jsonl', 'a') as f:
        f.write(json.dumps(result) + '\n\n')
