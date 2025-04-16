import os




if __name__ == "__main__":
    for filename in os.listdir("api"):
        if filename.endswith(".rst"):
            # line_prepender('api/' + filename, '.. title:: ' + filename[:-4])
            line_prepender("api/" + filename, filename[:-4])