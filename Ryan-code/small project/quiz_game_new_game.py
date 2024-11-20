def new_game():
    guesses = []
    correct_guesses = 0
    question_num = 1
    for key in question:
        print('-----------------------------')
        print(key)
        for i in option[question_num - 1]:
            print(i)
        guess = input('Enter  (A, B, C, or D)').upper()
        guesses.append(guess)
        correct_guesses += check_answer(question.get(key), guess)
        question_num += 1
    display_score(correct_guesses, guesses)
