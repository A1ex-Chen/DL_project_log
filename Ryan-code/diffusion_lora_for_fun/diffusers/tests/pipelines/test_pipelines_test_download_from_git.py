@slow
@require_torch_gpu
def test_download_from_git(self):
    clip_model_id = 'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'
    feature_extractor = CLIPImageProcessor.from_pretrained(clip_model_id)
    clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch
        .float16)
    pipeline = DiffusionPipeline.from_pretrained(
        'CompVis/stable-diffusion-v1-4', custom_pipeline=
        'clip_guided_stable_diffusion', clip_model=clip_model,
        feature_extractor=feature_extractor, torch_dtype=torch.float16)
    pipeline.enable_attention_slicing()
    pipeline = pipeline.to(torch_device)
    assert pipeline.__class__.__name__ == 'CLIPGuidedStableDiffusion'
    image = pipeline('a prompt', num_inference_steps=2, output_type='np'
        ).images[0]
    assert image.shape == (512, 512, 3)
