def get_sd_image(self, seed=0, shape=(4, 3, 512, 512), fp16=False):
    dtype = torch.float16 if fp16 else torch.float32
    image = torch.from_numpy(load_hf_numpy(self.get_file_format(seed, shape))
        ).to(torch_device).to(dtype)
    return image
