def evaluate(self, model):
    self.inferencer.inference(model, self.dataset)
    if self.dist_args['world_size'] > 1 and self.dist_args['global_rank'] != 0:
        while True:
            if not os.path.exists(os.path.join(self.save_base_dir, 'tmp')):
                break
            sleep(1.0)
        return None, None
    if self.dist_args['world_size'] > 1:
        results_path = self.merge_result_data()
    else:
        results_path = self.inferencer.results_path
    result = self.metric.metric(results_path)
    with open(os.path.join(self.save_base_dir, 'results.json'), 'w',
        encoding='utf-8') as f:
        f.write(json.dumps(dict(answer_path=results_path, result=result),
            indent=4))
    return results_path, result
