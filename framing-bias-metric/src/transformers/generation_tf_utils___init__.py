def __init__(self, num_beams, max_length, length_penalty, early_stopping):
    """
        Initialize n-best list of hypotheses.
        """
    self.max_length = max_length - 1
    self.length_penalty = length_penalty
    self.early_stopping = early_stopping
    self.num_beams = num_beams
    self.beams = []
    self.worst_score = 1000000000.0
