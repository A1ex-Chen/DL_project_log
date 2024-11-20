def on_fit_epoch_end(self, vals, epoch, best_fitness, fi):
    x = dict(zip(self.keys, vals))
    if self.csv:
        file = self.save_dir / 'results.csv'
        n = len(x) + 1
        s = '' if file.exists() else ('%20s,' * n % tuple(['epoch'] + self.
            keys)).rstrip(',') + '\n'
        with open(file, 'a') as f:
            f.write(s + ('%20.5g,' * n % tuple([epoch] + vals)).rstrip(',') +
                '\n')
    if self.tb:
        for k, v in x.items():
            self.tb.add_scalar(k, v, epoch)
    elif self.clearml:
        for k, v in x.items():
            title, series = k.split('/')
            self.clearml.task.get_logger().report_scalar(title, series, v,
                epoch)
    if self.wandb:
        if best_fitness == fi:
            best_results = [epoch] + vals[3:7]
            for i, name in enumerate(self.best_keys):
                self.wandb.wandb_run.summary[name] = best_results[i]
        self.wandb.log(x)
        self.wandb.end_epoch(best_result=best_fitness == fi)
    if self.clearml:
        self.clearml.current_epoch_logged_images = set()
        self.clearml.current_epoch += 1
