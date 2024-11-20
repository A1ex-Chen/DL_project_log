@staticmethod
def policy_simple():
    """Same as `policy_v0`, except with custom ops removed."""
    policy = [[('Color', 0.4, 9), ('Equalize', 0.6, 3)], [('Solarize', 0.8,
        3), ('Equalize', 0.4, 7)], [('Solarize', 0.4, 2), ('Solarize', 0.6,
        2)], [('Color', 0.2, 0), ('Equalize', 0.8, 8)], [('Equalize', 0.4, 
        8), ('SolarizeAdd', 0.8, 3)], [('Color', 0.6, 1), ('Equalize', 1.0,
        2)], [('Color', 0.4, 7), ('Equalize', 0.6, 0)], [('Posterize', 0.4,
        6), ('AutoContrast', 0.4, 7)], [('Solarize', 0.6, 8), ('Color', 0.6,
        9)], [('Equalize', 0.8, 4), ('Equalize', 0.0, 8)], [('Equalize', 
        1.0, 4), ('AutoContrast', 0.6, 2)], [('Posterize', 0.8, 2), (
        'Solarize', 0.6, 10)], [('Solarize', 0.6, 8), ('Equalize', 0.6, 1)]]
    return policy
