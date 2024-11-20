@staticmethod
def sorted(checkpoint_infos: List['Trainer.Callback.CheckpointInfo']) ->List[
    'Trainer.Callback.CheckpointInfo']:
    checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x.avg_loss)
    checkpoint_infos = sorted(checkpoint_infos, key=lambda x: x.accuracy,
        reverse=True)
    return checkpoint_infos
