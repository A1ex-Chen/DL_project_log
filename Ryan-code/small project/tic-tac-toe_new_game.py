def new_game():
    global player
    player = random.choice(players)
    label.config(text=player + ' turn')
    for row in range(3):
        for col in range(3):
            buttons[row][col].config(text='', bg='#F0F0F0')
