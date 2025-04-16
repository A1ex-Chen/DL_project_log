def display_table(rows, positions):

    def display_row(objects, positions):
        line = ''
        for i in range(len(objects)):
            line += str(objects[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)
    for objects in rows:
        display_row(objects, positions)
