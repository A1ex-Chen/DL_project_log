def compute_class_weights(self, weight_mode='median_frequency', c=1.02):
    assert weight_mode in ['median_frequency', 'logarithmic', 'linear']
    class_weighting_filepath = os.path.join(self.source_path,
        f'weighting_{weight_mode}_1+{self.n_classes_without_void}')
    if weight_mode == 'logarithmic':
        class_weighting_filepath += f'_c={c}'
    class_weighting_filepath += f'_{self.split}.pickle'
    if os.path.exists(class_weighting_filepath):
        class_weighting = pickle.load(open(class_weighting_filepath, 'rb'))
        print(f'Using {class_weighting_filepath} as class weighting')
        return class_weighting
    print('Compute class weights')
    n_pixels_per_class = np.zeros(self.n_classes)
    n_image_pixels_with_class = np.zeros(self.n_classes)
    for i in range(len(self)):
        label = self.load_label(i)
        h, w = label.shape
        current_dist = np.bincount(label.flatten(), minlength=self.n_classes)
        n_pixels_per_class += current_dist
        class_in_image = current_dist > 0
        n_image_pixels_with_class += class_in_image * h * w
        print(f'\r{i + 1}/{len(self)}', end='')
    print()
    n_pixels_per_class = n_pixels_per_class[1:]
    n_image_pixels_with_class = n_image_pixels_with_class[1:]
    if weight_mode == 'linear':
        class_weighting = n_pixels_per_class
    elif weight_mode == 'median_frequency':
        frequency = n_pixels_per_class / n_image_pixels_with_class
        class_weighting = np.median(frequency) / frequency
    elif weight_mode == 'logarithmic':
        probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
        class_weighting = 1 / np.log(c + probabilities)
    if np.isnan(np.sum(class_weighting)):
        print(f'n_pixels_per_class: {n_pixels_per_class}')
        print(f'n_image_pixels_with_class: {n_image_pixels_with_class}')
        print(f'class_weighting: {class_weighting}')
        raise ValueError('class weighting contains NaNs')
    with open(class_weighting_filepath, 'wb') as f:
        pickle.dump(class_weighting, f)
    print(f'Saved class weights under {class_weighting_filepath}.')
    return class_weighting
