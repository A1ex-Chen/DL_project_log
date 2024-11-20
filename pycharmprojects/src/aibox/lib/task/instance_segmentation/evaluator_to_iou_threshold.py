def to_iou_threshold(self) ->float:
    if self == Evaluator.Evaluation.Quality.LOOSEST:
        return 0.05
    elif self == Evaluator.Evaluation.Quality.LOOSE:
        return 0.25
    elif self == Evaluator.Evaluation.Quality.STANDARD:
        return 0.5
    elif self == Evaluator.Evaluation.Quality.STRICT:
        return 0.75
    elif self == Evaluator.Evaluation.Quality.STRICTEST:
        return 0.95
    else:
        raise ValueError('Invalid quality')
