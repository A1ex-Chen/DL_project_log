def game_over():
    canvas.delete(ALL)
    canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
        font=('consolas', 70), text='GAME OVER', fill='red')
