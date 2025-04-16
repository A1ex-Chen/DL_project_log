def print_errors(self, Error_stats):
    """ print various statistcs of Linf errors """
    print()
    print('conversion correctness test results')
    print('-----------------------------------')
    import pandas as pd
    print(pd.DataFrame(Error_stats))
