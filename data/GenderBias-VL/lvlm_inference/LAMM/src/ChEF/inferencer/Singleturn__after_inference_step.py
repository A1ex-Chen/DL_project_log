def _after_inference_step(self, predictions):
    if self.dist_args['world_size'] == 1:
        time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        answer_path = os.path.join(self.save_base_dir,
            f'{self.dataset_name}_{time}.json')
    else:
        base_dir = os.path.join(self.save_base_dir, 'tmp')
        os.makedirs(base_dir, exist_ok=True)
        global_rank = self.dist_args['global_rank']
        answer_path = os.path.join(base_dir,
            f'{self.dataset_name}_{global_rank}.json')
    with open(answer_path, 'w', encoding='utf8') as f:
        f.write(json.dumps(predictions, indent=4, ensure_ascii=False))
    self.results_path = answer_path
