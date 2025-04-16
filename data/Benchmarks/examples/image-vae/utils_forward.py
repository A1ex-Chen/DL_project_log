def forward(self, img1, img2):
    return self.ms_ssim(img1, img2)
