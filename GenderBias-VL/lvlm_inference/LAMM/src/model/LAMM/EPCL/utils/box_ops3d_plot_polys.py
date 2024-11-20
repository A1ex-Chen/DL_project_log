def plot_polys(plist, scale=500.0):
    fig, ax = plt.subplots()
    patches = []
    for p in plist:
        poly = Polygon(np.array(p) / scale, True)
        patches.append(poly)
