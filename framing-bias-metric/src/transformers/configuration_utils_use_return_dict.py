@property
def use_return_dict(self) ->bool:
    """
        :obj:`bool`: Whether or not return :class:`~transformers.file_utils.ModelOutput` instead of tuples.
        """
    return self.return_dict and not self.torchscript
