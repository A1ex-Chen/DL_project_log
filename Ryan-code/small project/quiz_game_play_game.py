def play_game():
    response = input('Do you want to play again? (yes or no): ').upper()
    if response == 'YES':
        return True
    else:
        return False
