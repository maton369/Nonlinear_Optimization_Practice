"""
コード2.1 ■ バックトラッキング法（アルゴリズム2.2）のコード
このコードは、非線形最適化における「線探索法（Line Search）」の一種である
バックトラッキング法（Armijo条件に基づく）を実装している。
目的関数の値が十分に減少するステップ幅 α を探索する手法である。
"""

import numpy as np


def line_Armijo(obj_f, nab_f, x_k, d_k, alpha=1, tau=0.5, sig1=0.0001):
    """
    Armijoのバックトラッキング線探索法を用いて最適なステップ幅を求める関数。

    Parameters
    ----------
    obj_f : function
        最小化したい目的関数 f(x)
    nab_f : function
        目的関数の勾配（∇f(x)）を返す関数
    x_k : ndarray
        現在の点（探索中のパラメータベクトル）
    d_k : ndarray
        探索方向ベクトル（通常は最急降下方向や共役勾配方向など）
    alpha : float, optional
        初期ステップ幅（デフォルト: 1）
    tau : float, optional
        ステップ幅の縮小率。0 < τ < 1（デフォルト: 0.5）
    sig1 : float, optional
        Armijo条件の閾値（減少率係数）。0 < σ₁ < 1（デフォルト: 0.0001）

    Returns
    -------
    alpha : float
        Armijo条件を満たす最終的なステップ幅
    """

    # 現在点と次の候補点の目的関数値を計算
    f_old = obj_f(x_k)  # 現在の点の関数値 f(x_k)
    f_new = obj_f(x_k + alpha * d_k)  # 次の点の関数値 f(x_k + αd_k)

    # 勾配と方向ベクトルの内積（方向微係数）を計算
    # これは、f(x_k) が d_k 方向にどれだけ変化するかを示す値
    nab_fTd = nab_f(x_k) @ d_k

    # Armijo条件を満たすまでステップ幅 α を縮小する
    # Armijo条件: f(x_k + αd_k) ≤ f(x_k) + σ₁α∇f(x_k)ᵀd_k
    while f_new > f_old + sig1 * alpha * nab_fTd:
        alpha = tau * alpha  # ステップ幅を τ 倍に縮小
        f_new = obj_f(x_k + alpha * d_k)  # 新しいステップ幅で関数値を再評価

    # 条件を満たしたら、その α を返す
    return alpha
