def __getitem__(self, i) ->Dict[str, torch.Tensor]:
    sources = self.list_data_dict[i]
    if isinstance(i, int):
        sources = [sources]
    assert len(sources) == 1, "Don't know why it is wrapped to a list"
    if 'image' in sources[0]:
        image_file = self.list_data_dict[i]['image']
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        image = Image.open(os.path.join(image_folder, image_file)).convert(
            'RGB')
        if self.data_args.image_aspect_ratio == 'pad':

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width),
                        background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height),
                        background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result
            image = expand2square(image, tuple(int(x * 255) for x in
                processor.image_mean))
            image = processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
        else:
            image = processor.preprocess(image, return_tensors='pt')[
                'pixel_values'][0]
        sources = preprocess_multimodal(copy.deepcopy([e['conversations'] for
            e in sources]), self.data_args)
    else:
        sources = copy.deepcopy([e['conversations'] for e in sources])
    data_dict = preprocess(sources, self.tokenizer, has_image='image' in
        self.list_data_dict[i])
    if isinstance(i, int):
        data_dict = dict(input_ids=data_dict['input_ids'][0], labels=
            data_dict['labels'][0])
    if 'image' in self.list_data_dict[i]:
        data_dict['images'] = image
    elif self.data_args.is_multimodal:
        crop_size = self.data_args.image_processor.crop_size
        data_dict['images'] = torch.zeros(3, crop_size['height'], crop_size
            ['width'])
    return data_dict
