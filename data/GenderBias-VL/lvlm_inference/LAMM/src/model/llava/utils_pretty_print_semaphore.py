def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return 'None'
    return f'Semaphore(value={semaphore._value}, locked={semaphore.locked()})'
