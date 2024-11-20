def init_inference(self, bsz):
    """ 
        init_inference()
        """
    for rnn in self.rnns:
        rnn.init_inference(bsz)
