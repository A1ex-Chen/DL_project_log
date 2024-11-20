@abstractmethod
def tc(self):
    """
		Tensor core usage by the kernel.
		Return "1" (yes), "0" (no, but possible), "-" (not applicable)
		"""
    pass
