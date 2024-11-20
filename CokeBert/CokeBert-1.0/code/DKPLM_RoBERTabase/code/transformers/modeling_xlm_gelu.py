def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/transformers/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
