@staticmethod
def policy_test():
    """Autoaugment test policy for debugging."""
    policy = [[('TranslateX', 1.0, 4), ('Equalize', 1.0, 10)]]
    return policy
