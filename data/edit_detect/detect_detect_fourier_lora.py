def detect_fourier_lora():
    ts = 1
    n = 100
    eps_dis_num: int = 100
    batch_size: int = 1024
    model_id = (
        'VillanDiffusion/fres_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_sde_c1.0_p0.0_epr0.0_BOX_14-HAT_psi1_lr6e-05_vp1.0_ve1.0'
        )
    lora_id = (
        'VillanDiffusion/fres_DDPM-CELEBA-HQ-256_CELEBA-HQ_ep50_sde_c1.0_p0.0_epr0.0_BOX_14-HAT_psi1_lr6e-05_vp1.0_ve1.0/lora'
        )
    name_prefix = 'fourier_lora'
    real_images = 'real_images/celeba_hq_256_jpg_n2048'
    out_dist_real_images = 'fake_images/celeba_hq_256_ddpm'
    fake_images = 'fake_images/celeba_hq_256_ddpm_1'
    pipeline = DDPMPipeline.from_pretrained(model_id, low_cpu_mem_usage=
        False, device_map=None, local_files_only=True)
    pipeline.load_lora_weights(lora_id)
    plot_direct_reconst(name_prefix=name_prefix, pipeline=pipeline,
        real_images=real_images, out_dist_real_images=out_dist_real_images,
        fake_images=fake_images, sample_num=100, batch_size=1024, timestep=
        ts, n=n)
