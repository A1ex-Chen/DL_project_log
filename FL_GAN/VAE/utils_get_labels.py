def get_labels(dataset_name: str='stl'):
    if dataset_name == 'stl':
        label_names = ['airplane', 'bird', 'car', 'cat', 'deer', 'dog',
            'horse', 'monkey', 'ship', 'truck']
    elif dataset_name == 'cifar':
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_name == 'fashion-mnist':
        label_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
            'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif dataset_name == 'mnist':
        label_names = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six',
            'Seven', 'Eight', 'Nine']
    return label_names
