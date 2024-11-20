def create_motion_field_and_warp_latents(motion_field_strength_x,
    motion_field_strength_y, frame_ids, latents):
    """
    Creates translation motion and warps the latents accordingly

    Args:
        motion_field_strength_x: motion strength along x-axis
        motion_field_strength_y: motion strength along y-axis
        frame_ids: indexes of the frames the latents of which are being processed.
            This is needed when we perform chunk-by-chunk inference
        latents: latent codes of frames

    Returns:
        warped_latents: warped latents
    """
    motion_field = create_motion_field(motion_field_strength_x=
        motion_field_strength_x, motion_field_strength_y=
        motion_field_strength_y, frame_ids=frame_ids, device=latents.device,
        dtype=latents.dtype)
    warped_latents = latents.clone().detach()
    for i in range(len(warped_latents)):
        warped_latents[i] = warp_single_latent(latents[i][None],
            motion_field[i][None])
    return warped_latents
