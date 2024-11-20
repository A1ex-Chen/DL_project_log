def ms_ssim(self, img1, img2, levels=5):
    weight = Variable(torch.Tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        ).cuda(img1.get_device()))
    msssim = Variable(torch.Tensor(levels).cuda(img1.get_device()))
    mcs = Variable(torch.Tensor(levels).cuda(img1.get_device()))
    for i in range(levels):
        ssim_map, mcs_map = self._ssim(img1, img2)
        msssim[i] = ssim_map
        mcs[i] = mcs_map
        filtered_im1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
        filtered_im2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        img1 = filtered_im1
        img2 = filtered_im2
    value = torch.prod(mcs[0:levels - 1] ** weight[0:levels - 1]) * msssim[
        levels - 1] ** weight[levels - 1]
    return value
