def on_train_end(self, logs=None):
    logs = logs or {}
    run_end = datetime.now()
    run_duration = run_end - self.run_timestamp
    run_in_hour = run_duration.total_seconds() / (60 * 60)
    send = {'run_id': self.run_id, 'runtime_hours': {'set': run_in_hour},
        'end_time': {'set': str(run_end)}, 'status': {'set': 'Finished'},
        'date_modified': {'set': 'NOW'}}
    self.log_messages.append(send)
    self.save()
