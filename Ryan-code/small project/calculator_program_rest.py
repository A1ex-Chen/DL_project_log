from tkinter import *








window = Tk()
window.title("Calculator program")
window.geometry("500x550")

equation_text = ""

equation_label = StringVar()

label = Label(window, textvariable=equation_label, font=("consolas", 20), bg="white", width=24, height=2)
label.pack()

frame = Frame(window)
frame.pack()

number = [[1,2,3],
          [4,5,6],
          [7,8,9]]

count = -1
for row in number:
    count += 1
    for col in range(len(row)):
        button = Button(frame, text=row[col], height=4, width=9, font=35,
                        command=lambda r=row,c=col: button_press(r[c])) #用參數r和c捕捉當下按鈕的row col，使得按下時抓取到對的數值，不然都會抓到最後一項
        button.grid(row=count, column=col)

button0 = Button(frame, text=0, height=4, width=9, font=35,
                 command=lambda: button_press(0))
button0.grid(row=3, column=1)

# button1 = Button(frame, text=1, height=4, width=9, font=35,
#                  command=lambda: button_press(1)) # 按下按鈕當下才執行
# button1.grid(row=0, column=0)
#
# button2 = Button(frame, text=2, height=4, width=9, font=35,
#                  command=lambda: button_press(2))
# button2.grid(row=0, column=1)
#
# button3 = Button(frame, text=3, height=4, width=9, font=35,
#                  command=lambda: button_press(3))
# button3.grid(row=0, column=2)
#
# button4 = Button(frame, text=4, height=4, width=9, font=35,
#                  command=lambda: button_press(4))
# button4.grid(row=1, column=0)
#
# button5 = Button(frame, text=5, height=4, width=9, font=35,
#                  command=lambda: button_press(5))
# button5.grid(row=1, column=1)
#
# button6 = Button(frame, text=6, height=4, width=9, font=35,
#                  command=lambda: button_press(6))
# button6.grid(row=1, column=2)
#
# button7 = Button(frame, text=7, height=4, width=9, font=35,
#                  command=lambda: button_press(7))
# button7.grid(row=2, column=0)
#
# button8 = Button(frame, text=8, height=4, width=9, font=35,
#                  command=lambda: button_press(8))
# button8.grid(row=2, column=1)
#
# button9 = Button(frame, text=9, height=4, width=9, font=35,
#                  command=lambda: button_press(9))
# button9.grid(row=2, column=2)
#
button0 = Button(frame, text=0, height=4, width=9, font=35,
                 command=lambda: button_press(0))
button0.grid(row=3, column=1)

plus = Button(frame, text='+', height=4, width=9, font=35,
              command=lambda: button_press("+"))
plus.grid(row=0, column=3)

minus = Button(frame, text='-', height=4, width=9, font=35,
               command=lambda: button_press("-"))
minus.grid(row=1, column=3)

multiply = Button(frame, text='*', height=4, width=9, font=35,
                  command=lambda: button_press("*"))
multiply.grid(row=2, column=3)

divide = Button(frame, text='/', height=4, width=9, font=35,
                command=lambda: button_press("/"))
divide.grid(row=3, column=3)

equal = Button(frame, text='=', height=4, width=9, font=35,
               command=equals)
equal.grid(row=3, column=2)

decimal = Button(frame, text='.', height=4, width=9, font=35,
                 command=lambda: button_press("."))
decimal.grid(row=3, column=0)

clear = Button(window, text='clear', height=4, width=12, font=35,
               command=clear)
clear.pack()

window.mainloop()