def get_window_size():
    window_width = window.winfo_width()
    window_height = window.winfo_height()
    screen_width = window.winfo_screenwidth()
    x = int(screen_width / 2 - window_width / 2)
    y = 0
    window.geometry(f'{window_width}x{window_height}+{x}+{y}')
