def next_turn(snake, food):
    x, y = snake.coordinates[0]
    if direction == 'up':
        y -= SPACE_SIZE
    elif direction == 'down':
        y += SPACE_SIZE
    elif direction == 'left':
        x -= SPACE_SIZE
    elif direction == 'right':
        x += SPACE_SIZE
    snake.coordinates.insert(0, (x, y))
    square = canvas.create_rectangle(x, y, x + SPACE_SIZE, y + SPACE_SIZE,
        fill=SNAKE_COLOR)
    snake.squares.insert(0, square)
    if x == food.coordinates[0] and y == food.coordinates[1]:
        global score
        score += 1
        label.config(text=f'Score:{score}')
        canvas.delete('food')
        food = Food()
    else:
        del snake.coordinates[-1]
        canvas.delete(snake.squares[-1])
        del snake.squares[-1]
    if check_collisions(snake):
        game_over()
        return
    else:
        window.after(SPEED, next_turn, snake, food)
