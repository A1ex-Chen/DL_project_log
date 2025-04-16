@require_torch
def test_torch_pytree(self):
    import torch
    import torch.utils._pytree
    data = np.random.rand(1, 3, 4, 4)
    x = CustomOutput(images=data)
    self.assertFalse(torch.utils._pytree._is_leaf(x))
    expected_flat_outs = [data]
    expected_tree_spec = torch.utils._pytree.TreeSpec(CustomOutput, [
        'images'], [torch.utils._pytree.LeafSpec()])
    actual_flat_outs, actual_tree_spec = torch.utils._pytree.tree_flatten(x)
    self.assertEqual(expected_flat_outs, actual_flat_outs)
    self.assertEqual(expected_tree_spec, actual_tree_spec)
    unflattened_x = torch.utils._pytree.tree_unflatten(actual_flat_outs,
        actual_tree_spec)
    self.assertEqual(x, unflattened_x)
