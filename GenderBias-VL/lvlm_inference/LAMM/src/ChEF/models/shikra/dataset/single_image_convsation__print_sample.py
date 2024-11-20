def _print_sample(self, ret_dict, raw_conv, conv):
    if not hasattr(self, '_printed_sample'):
        self._printed_sample = True
        post_processed_labels = post_process_generate_ids(self.preprocessor
            ['text'], ret_dict['labels'])
        print(f'=================== {self.mode} sample ===================',
            flush=True)
        print(
            f"        input_ids: {self.preprocessor['text'].convert_ids_to_tokens(ret_dict['input_ids'])}"
            )
        print(
            f"           labels: {self.preprocessor['text'].convert_ids_to_tokens(post_processed_labels)}"
            )
        print(
            f"decoded input_ids: {self.preprocessor['text'].decode(ret_dict['input_ids'])}"
            )
        print(
            f"decoded    labels: {self.preprocessor['text'].decode(post_processed_labels)}"
            )
        if 'image' in ret_dict and ret_dict['image'] is not None:
            image = ret_dict['image']
            if isinstance(image, torch.Tensor):
                print(f'            image: {image.shape}')
            elif isinstance(image, dict):
                print(f'            image: {image.keys()}')
            elif isinstance(image, list) and len(image) > 0:
                print(f'            image: {len(image)}, {type(image[0])}')
            else:
                print(f'            image: {type(image)}')
        print('====================================================', flush
            =True)
        try:
            if self.training_args is not None:
                _save_obj = {'ret_dict': ret_dict, 'raw_conv': raw_conv,
                    'conv': conv.get_prompt()}
                from pathlib import Path
                output_dir = Path(self.training_args.output_dir)
                output_dir.mkdir(exist_ok=True, parents=True)
                _local_rank = self.training_args.local_rank
                _word_size = self.training_args.world_size
                _file_path = str(output_dir /
                    f'sample_check_{self.mode}_{_local_rank}_{_word_size}.pt')
                print(f'saving some sample to {_file_path} for check.')
                torch.save(_save_obj, _file_path)
        except Exception as e:
            warnings.warn(
                f'try to save samples but get exception: {e.args}. ignored.')
