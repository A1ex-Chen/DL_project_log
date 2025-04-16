def compute_metrics(self, metrics=None, generate_overall=True):
    if metrics is None:
        metrics = list(mm.metrics.motchallenge_metrics)
    self.summary_text, self.summary_dataframe = eval_motChallenge(
        matrixes_predictions=self.matrixes_predictions, paths=self.paths,
        metrics=metrics, generate_overall=generate_overall)
    return self.summary_dataframe
