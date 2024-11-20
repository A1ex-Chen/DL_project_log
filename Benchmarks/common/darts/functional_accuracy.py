def accuracy(target: torch.tensor, output: torch.tensor):
    """Computes accuracy

    Args:
        output: logits of the model
        target: true labels

    Returns:
        accuracy of the predictions
    """
    return output.argmax(1).eq(target).double().mean().item()
