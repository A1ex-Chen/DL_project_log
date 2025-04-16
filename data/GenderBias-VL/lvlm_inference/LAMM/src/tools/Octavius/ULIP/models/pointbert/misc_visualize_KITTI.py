def visualize_KITTI(path, data_list, titles=['input', 'pred'], cmap=['bwr',
    'autumn'], zdir='y', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1)):
    fig = plt.figure(figsize=(6 * len(data_list), 6))
    cmax = data_list[-1][:, 0].max()
    for i in range(len(data_list)):
        data = data_list[i][:-2048] if i == 1 else data_list[i]
        color = data[:, 0] / cmax
        ax = fig.add_subplot(1, len(data_list), i + 1, projection='3d')
        ax.view_init(30, -120)
        b = ax.scatter(data[:, 0], data[:, 1], data[:, 2], zdir=zdir, c=
            color, vmin=-1, vmax=1, cmap=cmap[0], s=4, linewidth=0.05,
            edgecolors='black')
        ax.set_title(titles[i])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.2, hspace=0)
    if not os.path.exists(path):
        os.makedirs(path)
    pic_path = path + '.png'
    fig.savefig(pic_path)
    np.save(os.path.join(path, 'input.npy'), data_list[0].numpy())
    np.save(os.path.join(path, 'pred.npy'), data_list[1].numpy())
    plt.close(fig)
