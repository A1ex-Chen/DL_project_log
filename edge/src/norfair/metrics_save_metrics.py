def save_metrics(self, save_path='.', file_name='metrics.txt'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    metrics_path = os.path.join(save_path, file_name)
    metrics_file = open(metrics_path, 'w+')
    metrics_file.write(self.summary_text)
    metrics_file.close()
