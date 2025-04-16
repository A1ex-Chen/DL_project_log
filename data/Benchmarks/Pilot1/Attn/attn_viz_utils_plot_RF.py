def plot_RF(recall_keras, precision_keras, pr_keras, no_skill, fname,
    xlabel_add='', ylabel_add=''):
    plt.figure()
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(recall_keras, precision_keras, label=
        'PR Keras (area = {:.3f})'.format(pr_keras))
    plt.xlabel('Recall' + xlabel_add)
    plt.ylabel('Precision' + ylabel_add)
    plt.title('PR curve')
    plt.legend(loc='best')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
