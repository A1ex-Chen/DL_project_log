@torch.jit.export
def reorder_encoder_out(self, encoder_outs: Optional[List[Dict[str, List[
    Tensor]]]], new_order):
    """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
    new_outs: List[Dict[str, List[Tensor]]] = []
    if not self.has_encoder():
        return new_outs
    for i, model in enumerate(self.models):
        assert encoder_outs is not None
        new_outs.append(model.encoder.reorder_encoder_out(encoder_outs[i],
            new_order))
    return new_outs
