def assert_instances_allclose(input, other, *, rtol=1e-05, msg='',
    size_as_tensor=False):
    """
    Args:
        input, other (Instances):
        size_as_tensor: compare image_size of the Instances as tensors (instead of tuples).
             Useful for comparing outputs of tracing.
    """
    if not isinstance(input, Instances):
        input = convert_scripted_instances(input)
    if not isinstance(other, Instances):
        other = convert_scripted_instances(other)
    if not msg:
        msg = 'Two Instances are different! '
    else:
        msg = msg.rstrip() + ' '
    size_error_msg = (msg +
        f'image_size is {input.image_size} vs. {other.image_size}!')
    if size_as_tensor:
        assert torch.equal(torch.tensor(input.image_size), torch.tensor(
            other.image_size)), size_error_msg
    else:
        assert input.image_size == other.image_size, size_error_msg
    fields = sorted(input.get_fields().keys())
    fields_other = sorted(other.get_fields().keys())
    assert fields == fields_other, msg + f'Fields are {fields} vs {fields_other}!'
    for f in fields:
        val1, val2 = input.get(f), other.get(f)
        if isinstance(val1, (Boxes, ROIMasks)):
            assert torch.allclose(val1.tensor, val2.tensor, atol=100 * rtol
                ), msg + f'Field {f} differs too much!'
        elif isinstance(val1, torch.Tensor):
            if val1.dtype.is_floating_point:
                mag = torch.abs(val1).max().cpu().item()
                assert torch.allclose(val1, val2, atol=mag * rtol
                    ), msg + f'Field {f} differs too much!'
            else:
                assert torch.equal(val1, val2
                    ), msg + f'Field {f} is different!'
        else:
            raise ValueError(f"Don't know how to compare type {type(val1)}")
