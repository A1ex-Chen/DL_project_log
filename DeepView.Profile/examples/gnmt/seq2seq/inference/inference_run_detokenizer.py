def run_detokenizer(self, eval_path):
    """
        Executes moses detokenizer on eval_path file and saves result to
        eval_path + ".detok" file.

        :param eval_path: path to the tokenized input
        """
    logging.info('Running detokenizer')
    detok_path = os.path.join(self.dataset_dir, config.DETOKENIZER)
    detok_eval_path = eval_path + '.detok'
    with open(detok_eval_path, 'w') as detok_eval_file, open(eval_path, 'r'
        ) as eval_file:
        subprocess.run(['perl', f'{detok_path}'], stdin=eval_file, stdout=
            detok_eval_file, stderr=subprocess.DEVNULL)
