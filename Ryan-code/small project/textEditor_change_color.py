def change_color():
    color = colorchooser.askcolor(title='pick a color...or else')
    text_area.config(fg=str(color[1]))
