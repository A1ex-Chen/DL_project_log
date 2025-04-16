@staticmethod
def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) ->torch.Tensor:
    """Spherical Linear intERPolation

        Args:
            x0 (`torch.Tensor`): first tensor to interpolate between
            x1 (`torch.Tensor`): seconds tensor to interpolate between
            alpha (`float`): interpolation between 0 and 1

        Returns:
            `torch.Tensor`: interpolated tensor
        """
    theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.
        norm(x0) / torch.norm(x1))
    return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta
        ) * x1 / sin(theta)
