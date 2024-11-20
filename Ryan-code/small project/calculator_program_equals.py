def equals():
    global equation_text
    try:
        total = str(eval(equation_text))
        equation_label.set(total)
        equation_text = total
    except SyntaxError:
        equation_label.set('Syntax error')
        equation_text = ''
    except ZeroDivisionError:
        equation_label.set('arithmetic error')
        equation_text = ''
