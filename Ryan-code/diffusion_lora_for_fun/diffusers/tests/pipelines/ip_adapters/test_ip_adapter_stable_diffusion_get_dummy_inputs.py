def get_dummy_inputs(self, for_image_to_image=False, for_inpainting=False,
    for_sdxl=False, for_masks=False, for_instant_style=False):
    image = load_image(
        'https://user-images.githubusercontent.com/24734142/266492875-2d50d223-8475-44f0-a7c6-08b51cb53572.png'
        )
    if for_sdxl:
        image = image.resize((1024, 1024))
    input_kwargs = {'prompt': 'best quality, high quality',
        'negative_prompt':
        'monochrome, lowres, bad anatomy, worst quality, low quality',
        'num_inference_steps': 5, 'generator': torch.Generator(device='cpu'
        ).manual_seed(33), 'ip_adapter_image': image, 'output_type': 'np'}
    if for_image_to_image:
        image = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/vermeer.jpg'
            )
        ip_image = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/river.png'
            )
        if for_sdxl:
            image = image.resize((1024, 1024))
            ip_image = ip_image.resize((1024, 1024))
        input_kwargs.update({'image': image, 'ip_adapter_image': ip_image})
    elif for_inpainting:
        image = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/inpaint_image.png'
            )
        mask = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/mask.png'
            )
        ip_image = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/girl.png'
            )
        if for_sdxl:
            image = image.resize((1024, 1024))
            mask = mask.resize((1024, 1024))
            ip_image = ip_image.resize((1024, 1024))
        input_kwargs.update({'image': image, 'mask_image': mask,
            'ip_adapter_image': ip_image})
    elif for_masks:
        face_image1 = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl1.png'
            )
        face_image2 = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_girl2.png'
            )
        mask1 = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask1.png'
            )
        mask2 = load_image(
            'https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_mask_mask2.png'
            )
        input_kwargs.update({'ip_adapter_image': [[face_image1], [
            face_image2]], 'cross_attention_kwargs': {'ip_adapter_masks': [
            mask1, mask2]}})
    elif for_instant_style:
        composition_mask = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/1024_whole_mask.png'
            )
        female_mask = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_None_20240321125641_mask.png'
            )
        male_mask = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_None_20240321125344_mask.png'
            )
        background_mask = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter_6_20240321130722_mask.png'
            )
        ip_composition_image = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125152.png'
            )
        ip_female_style = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125625.png'
            )
        ip_male_style = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321125329.png'
            )
        ip_background = load_image(
            'https://huggingface.co/datasets/OzzyGT/testing-resources/resolve/main/ip_adapter__20240321130643.png'
            )
        input_kwargs.update({'ip_adapter_image': [ip_composition_image, [
            ip_female_style, ip_male_style, ip_background]],
            'cross_attention_kwargs': {'ip_adapter_masks': [[
            composition_mask], [female_mask, male_mask, background_mask]]}})
    return input_kwargs
