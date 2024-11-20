from tkinter import *
import random










window = Tk()
window.title("Tic-Tac-Toe")
players = ["x", "o"]
player = random.choice(players)

buttons = [[0, 0, 0],
           [0, 0, 0],
           [0, 0, 0]]

label = Label(window, text=player + " turn", font=("consolas", 40))
label.pack(side="top")

reset_button = Button(window, text="restart", font=("consolas", 20), command=new_game)
reset_button.pack(side="top")

frame = Frame(window)
frame.pack()

for row in range(3):
    for col in range(3):
        buttons[row][col] = Button(frame, text="", font=("consolas", 40),
                                   width=5, height=2, command=lambda r=row, c=col: next_turn(r, c))
        buttons[row][col].grid(row=row, column=col)

window.mainloop()