def generate_file_contents():
    outfile = open(GROUND_TRUTH_FILENAME, 'a')
    for file in os.listdir(PERSON_DIR):
        input_filename = os.fsdecode(file)
        line = '%s, %d, %d\n' % (input_filename, 2, 1)
        outfile.write(line)
    for file in os.listdir(NON_PERSON_DIR):
        input_filename = os.fsdecode(file)
        line = '%s, %d, %d\n' % (input_filename, 2, 0)
        outfile.write(line)
    outfile.flush()
    outfile.close()
