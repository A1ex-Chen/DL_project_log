def __call__(self, model=None, iterations=10, cleanup=True):
    """
        Executes the hyperparameter evolution process when the Tuner instance is called.

        This method iterates through the number of iterations, performing the following steps in each iteration:
        1. Load the existing hyperparameters or initialize new ones.
        2. Mutate the hyperparameters using the `mutate` method.
        3. Train a YOLO model with the mutated hyperparameters.
        4. Log the fitness score and mutated hyperparameters to a CSV file.

        Args:
           model (Model): A pre-initialized YOLO model to be used for training.
           iterations (int): The number of generations to run the evolution for.
           cleanup (bool): Whether to delete iteration weights to reduce storage space used during tuning.

        Note:
           The method utilizes the `self.tune_csv` Path object to read and log hyperparameters and fitness scores.
           Ensure this path is set correctly in the Tuner instance.
        """
    t0 = time.time()
    best_save_dir, best_metrics = None, None
    (self.tune_dir / 'weights').mkdir(parents=True, exist_ok=True)
    for i in range(iterations):
        mutated_hyp = self._mutate()
        LOGGER.info(
            f'{self.prefix}Starting iteration {i + 1}/{iterations} with hyperparameters: {mutated_hyp}'
            )
        metrics = {}
        train_args = {**vars(self.args), **mutated_hyp}
        save_dir = get_save_dir(get_cfg(train_args))
        weights_dir = save_dir / 'weights'
        try:
            cmd = ['yolo', 'train', *(f'{k}={v}' for k, v in train_args.
                items())]
            return_code = subprocess.run(cmd, check=True).returncode
            ckpt_file = weights_dir / ('best.pt' if (weights_dir /
                'best.pt').exists() else 'last.pt')
            metrics = torch.load(ckpt_file)['train_metrics']
            assert return_code == 0, 'training failed'
        except Exception as e:
            LOGGER.warning(
                f"""WARNING ❌️ training failure for hyperparameter tuning iteration {i + 1}
{e}"""
                )
        fitness = metrics.get('fitness', 0.0)
        log_row = [round(fitness, 5)] + [mutated_hyp[k] for k in self.space
            .keys()]
        headers = '' if self.tune_csv.exists() else ','.join(['fitness'] +
            list(self.space.keys())) + '\n'
        with open(self.tune_csv, 'a') as f:
            f.write(headers + ','.join(map(str, log_row)) + '\n')
        x = np.loadtxt(self.tune_csv, ndmin=2, delimiter=',', skiprows=1)
        fitness = x[:, 0]
        best_idx = fitness.argmax()
        best_is_current = best_idx == i
        if best_is_current:
            best_save_dir = save_dir
            best_metrics = {k: round(v, 5) for k, v in metrics.items()}
            for ckpt in weights_dir.glob('*.pt'):
                shutil.copy2(ckpt, self.tune_dir / 'weights')
        elif cleanup:
            shutil.rmtree(weights_dir, ignore_errors=True)
        plot_tune_results(self.tune_csv)
        header = f"""{self.prefix}{i + 1}/{iterations} iterations complete ✅ ({time.time() - t0:.2f}s)
{self.prefix}Results saved to {colorstr('bold', self.tune_dir)}
{self.prefix}Best fitness={fitness[best_idx]} observed at iteration {best_idx + 1}
{self.prefix}Best fitness metrics are {best_metrics}
{self.prefix}Best fitness model is {best_save_dir}
{self.prefix}Best fitness hyperparameters are printed below.
"""
        LOGGER.info('\n' + header)
        data = {k: float(x[best_idx, i + 1]) for i, k in enumerate(self.
            space.keys())}
        yaml_save(self.tune_dir / 'best_hyperparameters.yaml', data=data,
            header=remove_colorstr(header.replace(self.prefix, '# ')) + '\n')
        yaml_print(self.tune_dir / 'best_hyperparameters.yaml')
