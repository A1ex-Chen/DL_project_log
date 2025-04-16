def hota_final_scores(res):
    """Calculate final HOTA scores"""
    res['HOTA'] = np.mean(res['HOTA'])
    res['DetA'] = np.mean(res['DetA'])
    res['AssA'] = np.mean(res['AssA'])
    res['DetRe'] = np.mean(res['DetRe'])
    res['DetPr'] = np.mean(res['DetPr'])
    res['AssRe'] = np.mean(res['AssRe'])
    res['AssPr'] = np.mean(res['AssPr'])
    res['LocA'] = np.mean(res['LocA'])
    res['RHOTA'] = np.mean(res['RHOTA'])
    res['HOTA_TP'] = np.mean(res['HOTA_TP'])
    res['HOTA_FP'] = np.mean(res['HOTA_FP'])
    res['HOTA_FN'] = np.mean(res['HOTA_FN'])
