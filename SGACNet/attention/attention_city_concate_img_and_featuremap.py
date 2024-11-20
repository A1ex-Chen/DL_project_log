def concate_img_and_featuremap(img, feature_map, img_percent,
    feature_map_percent):
    heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
    heatmap = cv2.addWeighted(heatmap, feature_map_percent, img, img_percent, 0
        )
    return heatmap
