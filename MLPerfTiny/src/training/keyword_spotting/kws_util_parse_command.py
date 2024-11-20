def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.
        getenv('HOME'), 'data'), help=
        """      Where to download the speech training data to. Or where it is already saved.
      """
        )
    parser.add_argument('--bg_path', type=str, default=os.path.join(os.
        getenv('PWD')), help=
        """      Where to find background noise folder.
      """)
    parser.add_argument('--background_volume', type=float, default=0.1,
        help=
        """      How loud the background noise should be, between 0 and 1.
      """
        )
    parser.add_argument('--background_frequency', type=float, default=0.8,
        help=
        """      How many of the training samples have background noise mixed in.
      """
        )
    parser.add_argument('--silence_percentage', type=float, default=10.0,
        help=
        """      How much of the training data should be silence.
      """)
    parser.add_argument('--unknown_percentage', type=float, default=10.0,
        help=
        """      How much of the training data should be unknown words.
      """
        )
    parser.add_argument('--time_shift_ms', type=float, default=100.0, help=
        """      Range to randomly shift the training audio by in time.
      """
        )
    parser.add_argument('--sample_rate', type=int, default=16000, help=
        'Expected sample rate of the wavs')
    parser.add_argument('--clip_duration_ms', type=int, default=1000, help=
        'Expected duration in milliseconds of the wavs')
    parser.add_argument('--window_size_ms', type=float, default=30.0, help=
        'How long each spectrogram timeslice is')
    parser.add_argument('--window_stride_ms', type=float, default=20.0,
        help='How long each spectrogram timeslice is')
    parser.add_argument('--feature_type', type=str, default='mfcc', choices
        =['mfcc', 'lfbe', 'td_samples'], help=
        'Type of input features. Valid values: "mfcc" (default), "lfbe", "td_samples"'
        )
    parser.add_argument('--dct_coefficient_count', type=int, default=10,
        help='How many MFCC or log filterbank energy features')
    parser.add_argument('--epochs', type=int, default=36, help=
        'How many epochs to train')
    parser.add_argument('--num_train_samples', type=int, default=-1, help=
        'How many samples from the training set to use')
    parser.add_argument('--num_val_samples', type=int, default=-1, help=
        'How many samples from the validation set to use')
    parser.add_argument('--num_test_samples', type=int, default=-1, help=
        'How many samples from the test set to use')
    parser.add_argument('--batch_size', type=int, default=100, help=
        'How many items to train with at once')
    parser.add_argument('--num_bin_files', type=int, default=1000, help=
        'How many binary test files for benchmark runner to create')
    parser.add_argument('--bin_file_path', type=str, default=os.path.join(
        os.getenv('HOME'), 'kws_test_files'), help=
        """      Directory where plots of binary test files for benchmark runner are written.
      """
        )
    parser.add_argument('--model_architecture', type=str, default='ds_cnn',
        help='What model architecture to use')
    parser.add_argument('--run_test_set', type=bool, default=True, help=
        'In train.py, run model.eval() on test set if True')
    parser.add_argument('--saved_model_path', type=str, default=
        'trained_models/kws_model.h5', help=
        'In quantize.py, path to load pretrained model from; in train.py, destination for trained model'
        )
    parser.add_argument('--model_init_path', type=str, default=None, help=
        'Path to load pretrained model for evaluation or starting point for training'
        )
    parser.add_argument('--tfl_file_name', default=
        'trained_models/kws_model.tflite', help=
        'File name to which the TF Lite model will be saved (quantize.py) or loaded (eval_quantized_model)'
        )
    parser.add_argument('--learning_rate', type=float, default=1e-05, help=
        'Initial LR')
    parser.add_argument('--lr_sched_name', type=str, default=
        'step_function', help='lr schedule scheme name to be picked from lr.py'
        )
    parser.add_argument('--plot_dir', type=str, default='./plots', help=
        """      Directory where plots of accuracy vs Epochs are stored
      """
        )
    parser.add_argument('--target_set', type=str, default='test', help=
        """      For eval_quantized_model, which set to measure.
      """)
    Flags, unparsed = parser.parse_known_args()
    return Flags, unparsed
