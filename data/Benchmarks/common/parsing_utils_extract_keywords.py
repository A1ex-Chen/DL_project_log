def extract_keywords(lst_dict, kw):
    """Extract the value associated to a specific keyword in a list of dictionaries. Returns the list of values extracted from the keywords.

    Parameters
    ----------
    lst_dict : python list of dictionaries
       list to extract keywords from
    kw : string
       keyword to extract from dictionary
    """
    lst = [di[kw] for di in lst_dict]
    return lst
