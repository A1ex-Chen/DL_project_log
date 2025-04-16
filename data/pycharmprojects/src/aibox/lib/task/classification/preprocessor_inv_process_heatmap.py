@classmethod
def inv_process_heatmap(cls, process_dict: Dict[str, Any], heatmap: np.ndarray
    ) ->np.ndarray:
    if process_dict[cls.PROCESS_KEY_IS_TRAIN_OR_EVAL] or process_dict[cls.
        PROCESS_KEY_EVAL_CENTER_CROP_RATIO] == 1:
        return super().inv_process_heatmap(process_dict, heatmap)
    else:
        assert heatmap.ndim == 3
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)
        right_pad, bottom_pad = process_dict[cls.PROCESS_KEY_RIGHT_PAD
            ], process_dict[cls.PROCESS_KEY_BOTTOM_PAD]
        inv_heatmap = F.pad(input=heatmap, pad=[0, -right_pad, 0, -bottom_pad])
        center_crop_ratio = process_dict[cls.PROCESS_KEY_EVAL_CENTER_CROP_RATIO
            ]
        crop_margin_width = inv_heatmap.shape[2
            ] / center_crop_ratio - inv_heatmap.shape[2]
        crop_margin_height = inv_heatmap.shape[1
            ] / center_crop_ratio - inv_heatmap.shape[1]
        x_pad = int(crop_margin_width // 2)
        y_pad = int(crop_margin_height // 2)
        inv_heatmap = F.pad(input=inv_heatmap, pad=[x_pad, x_pad, y_pad, y_pad]
            )
        origin_size = process_dict[cls.PROCESS_KEY_ORIGIN_HEIGHT
            ], process_dict[cls.PROCESS_KEY_ORIGIN_WIDTH]
        inv_heatmap = F.interpolate(input=inv_heatmap.unsqueeze(dim=0),
            size=origin_size, mode='bilinear', align_corners=True).squeeze(dim
            =0)
        return inv_heatmap.permute(1, 2, 0).numpy()
