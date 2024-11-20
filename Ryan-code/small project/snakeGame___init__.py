def __init__(self):
    x = random.randint(0, int(GAME_WIDTH / SPACE_SIZE - 1)) * SPACE_SIZE
    y = random.randint(0, int(GAME_HEIGHT / SPACE_SIZE - 1)) * SPACE_SIZE
    self.coordinates = [x, y]
    canvas.create_oval(x, y, x + SPACE_SIZE, y + SPACE_SIZE, fill=
        FOOD_COLOR, tag='food')
    canvas.pack()
