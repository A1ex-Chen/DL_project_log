def end_epoch(self, best_result=False):
    if self.wandb_run:
        wandb.log(self.log_dict)
        self.log_dict = {}
        if self.result_artifact:
            train_results = wandb.JoinedTable(self.val_table, self.
                result_table, 'id')
            self.result_artifact.add(train_results, 'result')
            wandb.log_artifact(self.result_artifact, aliases=['latest', 
                'epoch ' + str(self.current_epoch), 'best' if best_result else
                ''])
            self.result_table = wandb.Table(['epoch', 'id', 'prediction',
                'avg_confidence'])
            self.result_artifact = wandb.Artifact('run_' + wandb.run.id +
                '_progress', 'evaluation')
