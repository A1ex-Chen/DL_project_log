@torch.jit.script
def lamb_kernel(param, grad, exp_avg, exp_avg_sq, beta1: float, beta2:
    float, step_size: float, eps: float, weight_decay: float):
    exp_avg = exp_avg * beta1 + (1 - beta1) * grad
    exp_avg_sq = exp_avg_sq * beta2 + (1 - beta2) * (grad * grad)
    adam_step = exp_avg / (exp_avg_sq.sqrt() + eps)
    adam_step = adam_step + weight_decay * param
    weight_norm = param.norm(p=2).clamp(0, 10)
    adam_norm = adam_step.norm(p=2)
    trust_ratio = weight_norm / (adam_norm + eps)
    trust_ratio = (weight_norm == 0.0) * 1.0 + (weight_norm != 0.0
        ) * trust_ratio
    trust_ratio = (adam_norm == 0.0) * 1.0 + (adam_norm != 0.0) * trust_ratio
    trust_ratio = trust_ratio.float()
    param = param - step_size * trust_ratio * adam_step
    return param, exp_avg, exp_avg_sq
