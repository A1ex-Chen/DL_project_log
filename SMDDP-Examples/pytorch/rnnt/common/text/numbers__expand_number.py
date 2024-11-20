def _expand_number(m):
    if int(m.group(0)[0]) == 0:
        return _inflect.number_to_words(m.group(0), andword='', group=1)
    num = int(m.group(0))
    if num > 1000 and num < 3000:
        if num == 2000:
            return 'two thousand'
        elif num > 2000 and num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2
                ).replace(', ', ' ')
    elif num > 1000000000 and num % 10000 != 0:
        return _inflect.number_to_words(num, andword='', group=1)
    else:
        return _inflect.number_to_words(num, andword='')
