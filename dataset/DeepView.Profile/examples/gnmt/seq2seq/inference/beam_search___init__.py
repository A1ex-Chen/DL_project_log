def __init__(self, model, beam_size=5, max_seq_len=100, cuda=False,
    len_norm_factor=0.6, len_norm_const=5, cov_penalty_factor=0.1):
    """
        Constructor for the SequenceGenerator.

        Beam search decoding supports coverage penalty and length
        normalization. For details, refer to Section 7 of the GNMT paper
        (https://arxiv.org/pdf/1609.08144.pdf).

        :param model: model which implements generate method
        :param beam_size: decoder beam size
        :param max_seq_len: maximum decoder sequence length
        :param cuda: whether to use cuda
        :param len_norm_factor: length normalization factor
        :param len_norm_const: length normalization constant
        :param cov_penalty_factor: coverage penalty factor
        """
    self.model = model
    self.cuda = cuda
    self.beam_size = beam_size
    self.max_seq_len = max_seq_len
    self.len_norm_factor = len_norm_factor
    self.len_norm_const = len_norm_const
    self.cov_penalty_factor = cov_penalty_factor
    self.batch_first = self.model.batch_first
