from tkinter import *
import time



window = Tk()

time_label = Label(window, font=("Arial", 50), fg="#00FF00", bg="black")
time_label.pack()

day_label = Label(window, font=("Ink Free", 25))
day_label.pack(side=LEFT)

date_label = Label(window, font=("Ink Free", 25))
date_label.pack(side=RIGHT)

update()
window.mainloop()