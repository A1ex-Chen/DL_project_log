def check_all_rank_done(self):
    for i in range(self.dist_args['world_size']):
        if not os.path.exists(os.path.join(self.save_base_dir, 'tmp',
            f'{self.dataset_name}_{i}.json')):
            return False
    return True
