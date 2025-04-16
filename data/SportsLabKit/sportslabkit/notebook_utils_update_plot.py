def update_plot(frame_number, frames, ax):
    ax.clear()
    im = ax.imshow(frames[frame_number], origin='upper', animated=True)
    return [im]
