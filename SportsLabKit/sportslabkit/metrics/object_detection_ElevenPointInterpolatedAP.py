def ElevenPointInterpolatedAP(rec: Any, prec: Any) ->list[Any]:
    """Calculate 11-point interpolated average precision.

    Args:
        rec (np.ndarray[np.float64]): recall array
        prec (np.ndarray[np.float64]): precision array

    Returns:
        Interp_ap_info (list[Any]): List containing information necessary for ap calculation
    """
    mrec = []
    for e in rec:
        mrec.append(e)
    mpre = []
    for e in prec:
        mpre.append(e)
    recallValues = np.linspace(0, 1, 11)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    for r in recallValues:
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    ap = sum(rhoInterp) / 11
    Interp_ap_info = [ap, rhoInterp, recallValid, None]
    return Interp_ap_info
