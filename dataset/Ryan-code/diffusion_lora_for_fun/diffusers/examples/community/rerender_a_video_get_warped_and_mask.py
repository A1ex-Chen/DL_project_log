@torch.no_grad()
def get_warped_and_mask(flow_model, image1, image2, image3=None,
    pixel_consistency=False, device=None):
    if image3 is None:
        image3 = image1
    padder = InputPadder(image1.shape, padding_factor=8)
    image1, image2 = padder.pad(image1[None].to(device), image2[None].to(
        device))
    results_dict = flow_model(image1, image2, attn_splits_list=[2],
        corr_radius_list=[-1], prop_radius_list=[-1], pred_bidir_flow=True)
    flow_pr = results_dict['flow_preds'][-1]
    fwd_flow = padder.unpad(flow_pr[0]).unsqueeze(0)
    bwd_flow = padder.unpad(flow_pr[1]).unsqueeze(0)
    fwd_occ, bwd_occ = forward_backward_consistency_check(fwd_flow, bwd_flow)
    if pixel_consistency:
        warped_image1 = flow_warp(image1, bwd_flow)
        bwd_occ = torch.clamp(bwd_occ + (abs(image2 - warped_image1).mean(
            dim=1) > 255 * 0.25).float(), 0, 1).unsqueeze(0)
    warped_results = flow_warp(image3, bwd_flow)
    return warped_results, bwd_occ, bwd_flow
