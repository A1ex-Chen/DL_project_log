def run_safety_checker(self, image, device, dtype):
    if self.safety_checker is not None:
        feature_extractor_input = self.image_processor.postprocess(image,
            output_type='pil')
        safety_checker_input = self.feature_extractor(feature_extractor_input,
            return_tensors='pt').to(device)
        image, nsfw_detected, watermark_detected = self.safety_checker(images
            =image, clip_input=safety_checker_input.pixel_values.to(dtype=
            dtype))
    else:
        nsfw_detected = None
        watermark_detected = None
        if hasattr(self, 'unet_offload_hook'
            ) and self.unet_offload_hook is not None:
            self.unet_offload_hook.offload()
    return image, nsfw_detected, watermark_detected
