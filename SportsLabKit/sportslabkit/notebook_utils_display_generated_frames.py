def display_generated_frames(frames):

    def update_plot(frame_number, frames, ax):
        ax.clear()
        im = ax.imshow(frames[frame_number], origin='upper', animated=True)
        return [im]
    fig, ax = plt.subplots(figsize=(12, 6))
    ani = FuncAnimation(fig, update_plot, frames=len(frames), fargs=(frames,
        ax), interval=200, blit=True)
    html = HTML(ani.to_jshtml())
    ipy_display(html)
    plt.close()
