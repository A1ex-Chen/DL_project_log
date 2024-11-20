def export_torchscript(self):
    """
        Export the model to a ``torch.jit.TracedModule`` by tracing.
        The returned object can be saved to a file by ``.save()``.

        Returns:
            torch.jit.TracedModule: a torch TracedModule
        """
    logger = logging.getLogger(__name__)
    logger.info('Tracing the model with torch.jit.trace ...')
    with torch.no_grad():
        return torch.jit.trace(self.traceable_model, (self.traceable_inputs,))
