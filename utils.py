# coding: utf-8
import numpy as np

def generate_numvec(digit, z = None):
    # +2 は潜在空間の次元分
    out = np.zeros((1, 12))
    out[:, digit + 2] = 1.

    if z is None:
        return out
    else:
        # サンプリング空間が指定されているときに，その情報を落とし込んでおく
        for i in range(len(z)):
            out[:,i] = z[i]
        return out