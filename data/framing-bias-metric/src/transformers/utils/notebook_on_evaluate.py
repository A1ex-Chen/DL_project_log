def on_evaluate(self, args, state, control, metrics=None, **kwargs):
    if self.training_tracker is not None:
        values = {'Training Loss': 'No log'}
        for log in reversed(state.log_history):
            if 'loss' in log:
                values['Training Loss'] = log['loss']
                break
        if self.first_column == 'Epoch':
            values['Epoch'] = int(state.epoch)
        else:
            values['Step'] = state.global_step
        values['Validation Loss'] = metrics['eval_loss']
        _ = metrics.pop('total_flos', None)
        _ = metrics.pop('epoch', None)
        for k, v in metrics.items():
            if k == 'eval_loss':
                values['Validation Loss'] = v
            else:
                splits = k.split('_')
                name = ' '.join([part.capitalize() for part in splits[1:]])
                values[name] = v
        self.training_tracker.write_line(values)
        self.training_tracker.remove_child()
        self.prediction_bar = None
        self._force_next_update = True
