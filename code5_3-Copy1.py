# Gauss-Newton 法のシンプル実装である。
# 目的：非線形最小二乗問題
#     min_x  1/2 * || r(x) ||_2^2
# に対し、残差ベクトル r(x) とそのヤコビアン J(x) を用いて
# 近似ヘッセ B_k = J_k^T J_k と勾配 g_k = J_k^T r_k を構成し、
# 線形方程式 (J_k^T J_k) d_k = - J_k^T r_k を解いて更新するである。
#
# 数値的注意：
#  - Gauss-Newton は「残差が十分小さい」または「モデルがほぼ線形」のときに良好に収束するである。
#  - J_k^T J_k が特異/悪条件のとき np.linalg.solve が不安定になり得る。
#    その場合はダンピング（Levenberg-Marquardt：J^T J + λI）や正則化の併用が定石である。
#  - 収束判定としては ||J_k^T r_k||（= 勾配ノルム）だけでなく、||r_k|| の減少や ||d_k|| の小ささも併用すると堅牢である。

import numpy as np
from typing import Callable, Tuple


def gauss_newton(
    res_r: Callable[[np.ndarray], np.ndarray],  # 残差ベクトル r(x) を返す関数である
    jac_r: Callable[
        [np.ndarray], np.ndarray
    ],  # ヤコビアン J(x) を返す関数である（形状は (m, n) を想定）
    x_k: np.ndarray,  # 初期点ベクトルである（形状 (n,)）
    eps: float = 1e-6,  # 終了判定のしきい値（勾配ノルム基準）である
    max_iter: int = 1000,  # 最大反復回数である
) -> np.ndarray:
    """
    Gauss-Newton 法により非線形最小二乗問題を解くである。

    反復手順（k=0,1,...）：
      1) r_k = r(x_k), J_k = J(x_k) を計算
      2) 近似ヘッセ B_k = J_k^T J_k, 勾配 g_k = J_k^T r_k を構成
      3) (J_k^T J_k) d_k = - J_k^T r_k を解いて方向 d_k を得る
      4) x_{k+1} = x_k + d_k で更新
      5) ||g_k|| < eps なら停止

    Parameters
    ----------
    res_r : Callable[[np.ndarray], np.ndarray]
        r(x) を返す関数である（m 次元）。
    jac_r : Callable[[np.ndarray], np.ndarray]
        J(x) を返す関数である（形状 (m, n)）。
    x_k : np.ndarray
        初期点（n 次元）である。
    eps : float
        終了判定に用いる ||J^T r|| のしきい値である。
    max_iter : int
        反復の上限回数である。

    Returns
    -------
    np.ndarray
        近似解 x_k を返すである。
    """
    for k in range(max_iter):
        r_k = res_r(x_k)  # 残差ベクトル r(x_k) を計算するである
        J_k = jac_r(x_k)  # ヤコビアン J(x_k) を計算するである（形状 (m, n)）
        # Gauss-Newton の近似ヘッセと勾配を構成するである
        B_k = J_k.T @ J_k  # 近似ヘッセ B_k = J_k^T J_k である（形状 (n, n)）
        nab_f_k = J_k.T @ r_k  # 勾配 g_k = J_k^T r_k である（形状 (n,)）
        # 方向 d_k を線形方程式から求めるである
        d_k = np.linalg.solve(B_k, -nab_f_k)
        # 点を更新するである
        x_k = x_k + d_k
        # 勾配ノルム（= ||J^T r||）が十分小さければ停止するである
        if np.linalg.norm(nab_f_k) < eps:
            break

    # 収束情報の簡易出力である（総残差ノルムを併せて出力）
    print("GN, iter:", k + 1, "||r(x_k)||:", np.linalg.norm(r_k))
    return x_k
