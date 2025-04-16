def open_file():
    file = filedialog.askopenfilename(defaultextension='.txt', filetypes=[(
        'All Files', '*.*'), ('Text Document', '*.txt')])
    flag = True
    try:
        window.title(os.path.basename(file))
        text_area.delete(1.0, END)
        file = open(file, 'r')
        text_area.insert(1.0, file.read())
    except Exception:
        flag = False
        print('error!')
    finally:
        if flag:
            file.close()
