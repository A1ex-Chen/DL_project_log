def get_patch_layout(self, img_size):
    with torch.no_grad():
        dummy_img = torch.zeros([1] + img_size)
        dummy_out = self.proj(dummy_img)
    embed_dim = dummy_out.shape[1]
    patches_layout = tuple(dummy_out.shape[2:])
    num_patches = np.prod(patches_layout)
    return patches_layout, num_patches, embed_dim
