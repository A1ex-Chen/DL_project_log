def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2
