def list_to_vector_array(file_list, msg='calc...', n_mels=64, frames=5,
    n_fft=1024, hop_length=512, power=2.0):
    """
    convert the file_list to a vector array.
    file_to_vector_array() is iterated, and the output vector array is concatenated.

    file_list : list [ str ]
        .wav filename list of dataset
    msg : str ( default = "calc..." )
        description for tqdm.
        this parameter will be input into "desc" param at tqdm.

    return : numpy.array( numpy.array( float ) )
        vector array for training (this function is not used for test.)
        * dataset.shape = (number of feature vectors, dimensions of feature vectors)
    """
    dims = n_mels * frames
    for idx in tqdm(range(len(file_list)), desc=msg):
        vector_array = com.file_to_vector_array(file_list[idx], n_mels=
            n_mels, frames=frames, n_fft=n_fft, hop_length=hop_length,
            power=power)
        if idx == 0:
            dataset = numpy.zeros((vector_array.shape[0] * len(file_list),
                dims), float)
        dataset[vector_array.shape[0] * idx:vector_array.shape[0] * (idx + 
            1), :] = vector_array
    return dataset
