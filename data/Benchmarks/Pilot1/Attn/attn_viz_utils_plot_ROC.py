def plot_ROC(fpr_keras, tpr_keras, auc_keras, fname, xlabel_add='',
    ylabel_add='', zoom=False):
    plt.figure()
    if zoom:
        plt.xlim(0, 0.2)
        plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(
        auc_keras))
    plt.xlabel('False positive rate' + xlabel_add)
    plt.ylabel('True positive rate' + ylabel_add)
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
