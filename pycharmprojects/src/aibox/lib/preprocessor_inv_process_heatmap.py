@classmethod
def inv_process_heatmap(cls, process_dict: Dict[str, Any], heatmap: np.ndarray
    ) ->np.ndarray:
    assert heatmap.ndim == 3
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)
    right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD
        ], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
    inv_heatmap = F.pad(input=heatmap, pad=[0, -right_pad, 0, -bottom_pad])
    origin_size = process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT], process_dict[cls
        .PROCESS_KEY_ORIGIN_WIDTH]
    inv_heatmap = F.interpolate(input=inv_heatmap.unsqueeze(dim=0), size=
        origin_size, mode='bilinear', align_corners=True).squeeze(dim=0)
    return inv_heatmap.permute(1, 2, 0).numpy()
