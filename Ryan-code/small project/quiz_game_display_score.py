def display_score(correct_guesses, guesses):
    print('--------------------')
    print('RESULTS')
    print('--------------------')
    print('Answers: ', end='')
    for i in question:
        print(question.get(i), end=' ')
    print()
    print('Guesses: ', end='')
    for i in guesses:
        print(i, end=' ')
    print()
    score = int(correct_guesses / len(question) * 100)
    print('Your score is: ' + str(score) + '%')
