def test_only_cross_attention(self):
    torch.manual_seed(0)
    constructor_args = self.get_constructor_arguments(only_cross_attention=
        False)
    attn = Attention(**constructor_args)
    self.assertTrue(attn.to_k is not None)
    self.assertTrue(attn.to_v is not None)
    forward_args = self.get_forward_arguments(query_dim=constructor_args[
        'query_dim'], added_kv_proj_dim=constructor_args['added_kv_proj_dim'])
    self_and_cross_attn_out = attn(**forward_args)
    torch.manual_seed(0)
    constructor_args = self.get_constructor_arguments(only_cross_attention=True
        )
    attn = Attention(**constructor_args)
    self.assertTrue(attn.to_k is None)
    self.assertTrue(attn.to_v is None)
    forward_args = self.get_forward_arguments(query_dim=constructor_args[
        'query_dim'], added_kv_proj_dim=constructor_args['added_kv_proj_dim'])
    only_cross_attn_out = attn(**forward_args)
    self.assertTrue((only_cross_attn_out != self_and_cross_attn_out).all())
