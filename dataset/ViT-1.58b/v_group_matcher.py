@torch.jit.ignore
def group_matcher(self, coarse: bool=False) ->Dict:
    return dict(stem='^cls_token|pos_embed|patch_embed', blocks=[(
        '^blocks\\.(\\d+)', None), ('^norm', (99999,))])
