def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)
    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)
    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)
    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]
    return merged_img_ids, merged_eval_imgs
