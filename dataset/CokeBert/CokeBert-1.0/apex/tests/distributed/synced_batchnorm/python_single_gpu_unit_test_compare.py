def compare(desc, inp1, inp2, error):
    a = inp1.clone().detach().cpu().numpy()
    b = inp2.clone().detach().cpu().numpy()
    close = np.allclose(a, b, error, error)
    if not close:
        print(desc, close)
        z = a - b
        index = (np.abs(z) >= error + error * np.abs(b)).nonzero()
        print('dif    : ', z[index])
        print('inp1   : ', a[index])
        print('inp2   : ', b[index])
    return close
