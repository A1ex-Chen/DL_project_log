def update():
    time_string = time.strftime('%I : %M : %S %p')
    time_label.config(text=time_string)
    day_string = time.strftime('%A')
    day_label.config(text=day_string)
    date_string = time.strftime('%Y/%B/%d')
    date_label.config(text=date_string)
    window.after(1000, update)
