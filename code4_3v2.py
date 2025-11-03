"""
コード4.3 ■ BFGS 法のコード（点列のログを残すバージョン）
"""

import numpy as np
from code2_2 import *  # Wolfe条件のコードを読み込み
from code4_2 import *  # BFGS公式（H公式）のコードを読み込み


def QuasiNewton(obj_f, nab_f, x_k, max_iter=1000, eps=1.0e-8):
    seq_x = [x_k]
    H_k, nab_f_k = np.identity(len(x_k)), nab_f(x_k)  # 初期行列（単位行列）と勾配を計算
    for k in range(max_iter):
        d_k = -H_k @ nab_f_k  # 探索方向を計算
        alpha = line_Wolfe(obj_f, nab_f, x_k, d_k)  # 直線探索（Wolfe条件）
        x_k_old, nab_f_k_old = x_k, nab_f_k  # 前の点と勾配を保存
        x_k = x_k + alpha * d_k
        seq_x.append(x_k)
        nab_f_k = nab_f(x_k)
        if np.linalg.norm(nab_f_k) <= eps:
            break
        s_k, y_k = x_k - x_k_old, nab_f_k - nab_f_k_old  # s,yを計算
        H_k = BFGS_H(H_k, s_k, y_k)  # BFGS 公式による更新
    print("QNewton, iter:", k + 1, "f(x):", obj_f(x_k))
    return np.array(seq_x)
