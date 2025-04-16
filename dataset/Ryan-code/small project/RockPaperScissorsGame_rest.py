import random

my_list = ["rock", "paper", "scissors"]

while True:
    user = input("rock, paper, or scissors?: ")
    if user in list:
        computer = random.choice(my_list)
        print('computer : {}'.format(computer))
        print(f"player : {user}")
        if user == computer:
            print('Tie!')
        elif user == "rock":
            if computer == "paper":
                print("Computer Win!")
            elif computer == "scissors":
                print("User Win!")
        elif user == "paper":
            if computer == "scissors":
                print("Computer Win!")
            elif computer == "rock":
                print("User Win!")
        elif user == "scissors":
            if computer == "rock":
                print("Computer Win!")
            elif computer == "paper":
                print("User Win!")
        play_again = input("Yes or No? (input Yes or No): ").lower()
        if play_again != "Yes":
            break

print("Bye!")