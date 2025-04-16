def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / bar_num - 0.2, 1.03 *
            height, '%s' % float(height))
