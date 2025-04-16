def hash_list_of_strings(li):
    return str(abs(hash(''.join(li))))
