def plot(train_loss, valid_loss, accuracy):
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
    x_tran = range(0, len(train_loss), 1)
    x_valid = range(0, len(valid_loss), 1)
    x_acc = range(0, len(accuracy) * 3, 3)
    plt.figure()
    plt.plot(x_tran, train_loss, '-', color='red', label='VGG16 tarinning loss'
        )
    plt.plot(x_valid, valid_loss, ':', color='green', label='VGG16 Valid loss')
    plt.plot(x_acc, accuracy, '-.', color='blue', label='Accuracy')
    plt.title('the trainning for chicken', fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('the trainning for chicken', fontsize=24)
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('Score', fontsize=20)
    plt.legend(fontsize=16)
    plt.show()
