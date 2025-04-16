def visualize_feature_map_sum(item, line, name):
    """
    将每张子图进行相加
    :param feature_batch:
    :return:
    """
    feature_map = item.squeeze(0)
    c = item.shape[1]
    print(feature_map.shape)
    feature_map_combination = []
    for i in range(0, c):
        feature_map_split = feature_map.data.cpu().numpy()[i, :, :]
        feature_map_combination.append(feature_map_split)
    feature_map_sum = sum(one for one in feature_map_combination)
    plt.figure()
    plt.imshow(feature_map_sum, cmap='jet'),
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(
        '/home/yzhang/SGACNet/samples/feature/cityscapes_rgbd/feature/feature_map_sum_'
         + name + '_' + line + '.png')
    plt.show()
