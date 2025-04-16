@staticmethod
def backward(ctx, grad_hidden_states):
    grad_attn_output, grad_hidden_states = torch.chunk(grad_hidden_states, 
        2, dim=-1)
    attn_output, hidden_states = ctx.saved_tensors
    output = ReformerBackwardOutput(attn_output=attn_output, hidden_states=
        hidden_states, grad_attn_output=grad_attn_output,
        grad_hidden_states=grad_hidden_states)
    del grad_attn_output, grad_hidden_states, attn_output, hidden_states
    layers = ctx.layers
    all_buckets = ctx.all_buckets
    head_mask = ctx.head_mask
    attention_mask = ctx.attention_mask
    for idx, layer in enumerate(layers[::-1]):
        buckets = all_buckets[-1]
        all_buckets = all_buckets[:-1]
        output = layer.backward_pass(next_attn_output=output.attn_output,
            hidden_states=output.hidden_states, grad_attn_output=output.
            grad_attn_output, grad_hidden_states=output.grad_hidden_states,
            head_mask=head_mask[len(layers) - idx - 1], attention_mask=
            attention_mask, buckets=buckets)
    assert all_buckets == (), 'buckets have to be empty after backpropagation'
    grad_hidden_states = torch.cat([output.grad_attn_output, output.
        grad_hidden_states], dim=-1)
    return (grad_hidden_states, None, None, None, None, None, None, None,
        None, None, None, None)
