def _legacy_load_safety_checker(local_files_only, torch_dtype):
    from ..pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    feature_extractor = AutoImageProcessor.from_pretrained(
        'CompVis/stable-diffusion-safety-checker', local_files_only=
        local_files_only, torch_dtype=torch_dtype)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        'CompVis/stable-diffusion-safety-checker', local_files_only=
        local_files_only, torch_dtype=torch_dtype)
    return {'safety_checker': safety_checker, 'feature_extractor':
        feature_extractor}
