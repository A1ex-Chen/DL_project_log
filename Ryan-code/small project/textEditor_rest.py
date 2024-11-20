import os
from tkinter import *
from tkinter import filedialog, colorchooser, font
from tkinter.messagebox import *















window = Tk()
window.title("Text editor program")
file = None

window_width = 500
window_height = 500

# 獲取當前使用顯示器長和寬
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# 使window顯示在螢幕正中間，x和y分別只window左上角座標
x = int((screen_width / 2) - (window_width / 2))
y = int((screen_height / 2) - (window_height / 2))

# +{x}+{y}是指定window左上角的水平和垂直位置
window.geometry(f"{window_width}x{window_height}+{x}+{y}")


font_name = StringVar(window)
font_name.set("Arial")

font_size = StringVar(window)
font_size.set("25")

text_area = Text(window, font=(font_name.get(), font_size.get()))

# 在text_area裡創建一個 Scrollbar（捲軸）物件
scroll_bar = Scrollbar(text_area)

# 設定窗口的第一行（索引為0的列和行）的權重為1
# 權重可以隨著窗口變大變小去判斷誰要增多誰要減少
# ?????這一段不確定
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
text_area.grid(sticky=N + E + S + W)

scroll_bar.pack(side=RIGHT, fill=Y)
text_area.config(yscrollcommand=scroll_bar.set)

frame = Frame(window)
frame.grid()

color_button = Button(frame, text="color", command=change_color)
color_button.grid(row=0, column=0)

# OptionMenu(放在哪, 儲存OptionMenu的選擇, *font.families()此函數獲取所有可用字體的列表, command)
font_box = OptionMenu(frame, font_name, *font.families(), command=change_font)
font_box.grid(row=0, column=1)

size_box = Spinbox(frame, from_=1, to=100, textvariable=font_size, command=change_font)
size_box.grid(row=0, column=2)

menu_bar = Menu(window)
window.config(menu=menu_bar)

file_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)

file_menu.add_command(label="New", command=new_file)
file_menu.add_command(label="Open", command=open_file)
file_menu.add_command(label="Save", command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=quit)

edit_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="cut", command=cut)
edit_menu.add_command(label="Copy", command=copy)
edit_menu.add_command(label="Paste", command=paste)

help_menu = Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=about)


window.mainloop()


