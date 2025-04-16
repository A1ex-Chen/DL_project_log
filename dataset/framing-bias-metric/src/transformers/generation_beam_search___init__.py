def __init__(self, num_beams: int, max_length: int, length_penalty: float,
    early_stopping: bool):
    """
        Initialize n-best list of hypotheses.
        """
    self.max_length = max_length - 1
    self.length_penalty = length_penalty
    self.early_stopping = early_stopping
    self.num_beams = num_beams
    self.beams = []
    self.worst_score = 1000000000.0
