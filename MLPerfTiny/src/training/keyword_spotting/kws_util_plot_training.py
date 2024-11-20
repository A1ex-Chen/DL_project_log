def plot_training(plot_dir, history):
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.subplot(2, 1, 1)
    plt.plot(history.history['sparse_categorical_accuracy'], label=
        'Training Accuracy')
    plt.plot(history.history['val_sparse_categorical_accuracy'], label=
        'Val Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.savefig(plot_dir + '/acc.png')
