def button_press(num):
    global equation_text
    equation_text = equation_text + str(num)
    equation_label.set(equation_text)
