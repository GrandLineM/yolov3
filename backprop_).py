import numpy as np

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    dx, dw, db = None, None, None

    # Récupération des variables
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']

    # Initialisations
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # Dimensions
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_, W_ = dout.shape

    # db - dout (N, F, H', W')
    # On somme sur tous les éléments sauf les indices des filtres
    db = np.sum(dout, axis=(0, 2, 3))

    # dw = xp * dy
    # 0-padding juste sur les deux dernières dimensions de x
    xp = np.pad(x, ((0,), (0,), (pad,), (pad,)), 'constant')

    # Version sans vectorisation
    for n in range(N):  # On parcourt toutes les images
        for f in range(F):  # On parcourt tous les filtres
            for i in range(HH):  # indices du résultat
                for j in range(WW):
                    for k in range(H_):  # indices du filtre
                        for l in range(W_):
                            for c in range(C):  # profondeur
                                dw[f, c, i, j] += xp[n, c, stride * i + k, stride * j + l] * dout[n, f, k, l]

    # dx = dy_0 * w'
    # Valide seulement pour un stride = 1
    # 0-padding juste sur les deux dernières dimensions de dy = dout (N, F, H', W')
    doutp = np.pad(dout, ((0,), (0,), (WW - 1,), (HH - 1,)), 'constant')

    # 0-padding juste sur les deux dernières dimensions de dx
    dxp = np.pad(dx, ((0,), (0,), (pad,), (pad,)), 'constant')

    # filtre inversé dimension (F, C, HH, WW)
    w_ = np.zeros_like(w)
    for i in range(HH):
        for j in range(WW):
            w_[:, :, i, j] = w[:, :, HH - i - 1, WW - j - 1]

    # Version sans vectorisation
    for n in range(N):  # On parcourt toutes les images
        for f in range(F):  # On parcourt tous les filtres
            for i in range(H + 2 * pad):  # indices de l'entrée participant au résultat
                for j in range(W + 2 * pad):
                    for k in range(HH):  # indices du filtre
                        for l in range(WW):
                            for c in range(C):  # profondeur
                                dxp[n, c, i, j] += doutp[n, f, i + k, j + l] * w_[f, c, k, l]
    # Remove padding for dx
    dx = dxp[:, :, pad:-pad, pad:-pad]

    return dx, dw, db
    
print ()
