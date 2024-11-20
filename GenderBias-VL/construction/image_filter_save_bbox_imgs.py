def save_bbox_imgs(self, img, base_bboxes, base_phrases, save_dir, img_name):
    H, W = img.shape[1], img.shape[2]
    for i in range(base_bboxes.size(0)):
        base_bboxes[i] = base_bboxes[i] * torch.Tensor([W, H, W, H])
        base_bboxes[i][:2] -= base_bboxes[i][2:] / 2
        base_bboxes[i][2:] += base_bboxes[i][:2]
    base_bboxes = base_bboxes.cpu()
    img = inverse_normalize(img)
    img = Image.fromarray(np.uint8(img.transpose(1, 2, 0) * 255))
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    for box, label in zip(base_bboxes, base_phrases):
        show_box(box.numpy(), plt.gca(), label)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, img_name), bbox_inches='tight')
