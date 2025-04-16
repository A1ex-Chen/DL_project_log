def plot_study_txt(path='', x=None):
    fig, ax = plt.subplots(2, 4, figsize=(10, 6), tight_layout=True)
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), tight_layout=True)
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9],
            ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@.5', 'mAP@.5:.95', 't_inference (ms/img)',
            't_NMS (ms/img)', 't_total (ms/img)']
        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 100.0, '.-', linewidth=2,
            markersize=8, label=f.stem.replace('study_coco_', '').replace(
            'yolo', 'YOLO'))
    ax2.plot(1000.0 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 
        43.0, 47.5, 49.7, 51.5], 'k.-', linewidth=2, markersize=8, alpha=
        0.25, label='EfficientDet')
    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(str(Path(path).name) + '.png', dpi=300)
