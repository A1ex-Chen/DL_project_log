def _scale_confidence_score(score):
    """Scales the given confidence score by a factor specified in an environment variable."""
    scale = float(os.getenv('COMET_MAX_CONFIDENCE_SCORE', 100.0))
    return score * scale
