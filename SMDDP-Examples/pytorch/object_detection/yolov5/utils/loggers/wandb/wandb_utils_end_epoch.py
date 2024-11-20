def end_epoch(self, best_result=False):
    """
        commit the log_dict, model artifacts and Tables to W&B and flush the log_dict.

        arguments:
        best_result (boolean): Boolean representing if the result of this evaluation is best or not
        """
    if self.wandb_run:
        with all_logging_disabled():
            if self.bbox_media_panel_images:
                self.log_dict['BoundingBoxDebugger'
                    ] = self.bbox_media_panel_images
            try:
                wandb.log(self.log_dict)
            except BaseException as e:
                LOGGER.info(
                    f"""An error occurred in wandb logger. The training will proceed without interruption. More info
{e}"""
                    )
                self.wandb_run.finish()
                self.wandb_run = None
            self.log_dict = {}
            self.bbox_media_panel_images = []
        if self.result_artifact:
            self.result_artifact.add(self.result_table, 'result')
            wandb.log_artifact(self.result_artifact, aliases=['latest',
                'last', 'epoch ' + str(self.current_epoch), 'best' if
                best_result else ''])
            wandb.log({'evaluation': self.result_table})
            columns = ['epoch', 'id', 'ground truth', 'prediction']
            columns.extend(self.data_dict['names'])
            self.result_table = wandb.Table(columns)
            self.result_artifact = wandb.Artifact('run_' + wandb.run.id +
                '_progress', 'evaluation')
