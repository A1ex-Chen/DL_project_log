def next_turn(row, col):
    global player
    if buttons[row][col]['text'] == '' and check_winner() is False:
        if player == players[0]:
            buttons[row][col]['text'] = player
            if check_winner() is False:
                player = players[1]
                label.config(text=player + ' turn')
            elif check_winner() is True:
                label.config(text=player + ' wins')
            elif check_winner() == 'Tie':
                label.config(text='Tie!')
        else:
            buttons[row][col]['text'] = player
            if check_winner() is False:
                player = players[0]
                label.config(text=player + ' turn')
            elif check_winner() is True:
                label.config(text=player + ' wins')
            elif check_winner() == 'Tie':
                label.config(text='Tie!')
