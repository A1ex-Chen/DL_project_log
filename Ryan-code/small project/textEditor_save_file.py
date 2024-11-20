def save_file():
    file = filedialog.asksaveasfilename(initialfile='unititled.txt',
        defaultextension='.txt', filetypes=[('All Files', '*.*'), (
        'Text Documents', '*.txt')])
    flag = True
    if file is None:
        return
    else:
        try:
            window.title(os.path.basename(file))
            file = open(file, 'w')
            file.write(text_area.get(1.0, END))
        except Exception:
            flag = False
            print("couldn't save file")
        finally:
            if flag:
                file.close()
