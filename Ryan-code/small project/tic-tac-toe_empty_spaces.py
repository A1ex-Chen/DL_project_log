def empty_spaces():
    for row in range(3):
        for col in range(3):
            if buttons[row][col]['text'] == '':
                return True
    return False
