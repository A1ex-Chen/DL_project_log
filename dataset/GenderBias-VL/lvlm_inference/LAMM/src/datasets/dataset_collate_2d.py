def collate_2d(self, instances):
    vision_paths, output_texts, task_type = tuple([instance[key] for
        instance in instances] for key in ('vision_paths', 'output_texts',
        'task_type'))
    return dict(vision_paths=vision_paths, output_texts=output_texts,
        vision_type='image', task_type=task_type)
