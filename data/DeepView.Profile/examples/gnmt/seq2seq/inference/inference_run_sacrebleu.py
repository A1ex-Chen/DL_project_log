def run_sacrebleu(self, detok_eval_path, reference_path):
    """
        Executes sacrebleu and returns BLEU score.

        :param detok_eval_path: path to the test file
        :param reference_path: path to the reference file
        """
    if reference_path is None:
        reference_path = os.path.join(self.dataset_dir, config.
            TGT_TEST_TARGET_FNAME)
    sacrebleu_params = '--score-only -lc --tokenize intl'
    logging.info(f'Running sacrebleu (parameters: {sacrebleu_params})')
    sacrebleu = subprocess.run([
        f'sacrebleu --input {detok_eval_path}                                     {reference_path} {sacrebleu_params}'
        ], stdout=subprocess.PIPE, shell=True)
    test_bleu = float(sacrebleu.stdout.strip())
    return test_bleu
