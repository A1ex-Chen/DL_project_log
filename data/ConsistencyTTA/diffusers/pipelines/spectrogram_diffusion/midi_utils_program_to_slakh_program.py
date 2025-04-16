def program_to_slakh_program(program):
    for slakh_program in sorted(SLAKH_CLASS_PROGRAMS.values(), reverse=True):
        if program >= slakh_program:
            return slakh_program
