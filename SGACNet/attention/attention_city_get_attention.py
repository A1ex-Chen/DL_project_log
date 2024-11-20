def get_attention():
    root_path = '/home/yzhang/SGACNet/datasets/cityscapes/train/rgb/'
    base_path = (
        '/home/yzhang/SGACNet1/samples/feature/cityscapes_rgbd/context/')
    f = open(
        '/home/yzhang/SGACNet/samples/feature/cityscapes_rgbd/context/context_city.txt'
        , 'r')
    p = open('/home/yzhang/SGACNet/datasets/cityscapes/train_rgb1.txt', 'r')
    while True:
        line = f.readline()
        line = line[:-1]
        line1 = p.readline()
        line1 = line1[:-1]
        if line and line1:
            print(line)
            print(line1)
            rgb_path = root_path + line1
            feature_path = base_path + line
            img_rgb = cv_imread(rgb_path)
            img_rgb = process_img(img_rgb)
            img_feature = cv_imread(feature_path)
            img_feature = process_img(img_feature)
            img = concate_img_and_featuremap(img_rgb, img_feature, 0.3, 0.7)
            plt.figure()
            plt.imshow(img, cmap='jet')
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0,
                wspace=0)
            plt.margins(0, 0)
            plt.savefig(
                '/home/yzhang/SGACNet/samples/attention/city_context/' +
                '_' + line + '.jpg')
            plt.show()
        else:
            break
    f.close()
