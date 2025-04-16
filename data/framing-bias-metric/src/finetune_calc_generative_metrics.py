def calc_generative_metrics(self, preds, target) ->Dict:
    return calculate_rouge(preds, target)
