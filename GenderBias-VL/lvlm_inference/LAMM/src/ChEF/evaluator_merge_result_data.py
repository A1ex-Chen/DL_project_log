def merge_result_data(self):
    while True:
        sleep(1.0)
        if self.check_all_rank_done():
            sleep(1.0)
            all_results = []
            for i in range(self.dist_args['world_size']):
                with open(os.path.join(self.save_base_dir, 'tmp',
                    f'{self.dataset_name}_{i}.json'), 'r', encoding='utf8'
                    ) as f:
                    rank_result = json.load(f)
                    all_results += rank_result
            time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            answer_path = os.path.join(self.save_base_dir,
                f'{self.dataset_name}_{time}.json')
            with open(answer_path, 'w', encoding='utf8') as f:
                f.write(json.dumps(all_results, indent=4, ensure_ascii=False))
            shutil.rmtree(os.path.join(self.save_base_dir, 'tmp'))
            return answer_path
