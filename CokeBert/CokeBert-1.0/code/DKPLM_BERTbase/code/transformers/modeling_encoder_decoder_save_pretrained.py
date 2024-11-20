def save_pretrained(self, save_directory):
    """ Save a Seq2Seq model and its configuration file in a format such
        that it can be loaded using `:func:`~transformers.PreTrainedEncoderDecoder.from_pretrained`

        We save the encoder' and decoder's parameters in two separate directories.
        """
    self.encoder.save_pretrained(os.path.join(save_directory, 'encoder'))
    self.decoder.save_pretrained(os.path.join(save_directory, 'decoder'))
