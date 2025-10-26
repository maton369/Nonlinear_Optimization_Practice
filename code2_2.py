"""
コード2.2 ■ 挟み込みアルゴリズム（アルゴリズム2.3–アルゴリズム2.4）のコード
このコードは、非線形最適化の「Wolfe条件」を満たすステップ幅を求めるための
挟み込み（二分探索）アルゴリズムを実装している。
Armijo条件に加えて、勾配の変化も考慮する「強Wolfe条件」を用いる。
"""

import numpy as np


def line_Wolfe(obj_f, nab_f, x_k, d_k, alpha=1, alpha_max=5, sig1=0.0001, sig2=0.5):
    """
    Wolfe条件に基づく線探索を行う関数。

    Parameters
    ----------
    obj_f : function
        最小化したい目的関数 f(x)
    nab_f : function
        勾配（∇f(x)）を返す関数
    x_k : ndarray
        現在の探索点
    d_k : ndarray
        探索方向（通常は負の勾配方向など）
    alpha : float, optional
        初期ステップ幅（デフォルト: 1）
    alpha_max : float, optional
        最大ステップ幅（デフォルト: 5）
    sig1 : float, optional
        Armijo条件の閾値（デフォルト: 0.0001）
    sig2 : float, optional
        Wolfe条件の閾値（デフォルト: 0.5）

    Returns
    -------
    alpha : float
        Wolfe条件を満たすステップ幅
    """

    # 初期化
    alpha_old = 0  # 前回のステップ幅
    f0 = obj_f(x_k)  # 現在の関数値 f(x_k)
    nab_fTd = nab_f(x_k) @ d_k  # 勾配と方向ベクトルの内積（方向微係数）

    # -------------------------------------------------------------
    # 内部関数：Zoom アルゴリズム（アルゴリズム2.3）
    # 区間 [α_l, α_h] の中で Wolfe 条件を満たす α を探索する。
    # -------------------------------------------------------------
    def zoom(alpha_l, alpha_h, x_k, d_k, f_new, f_old):
        for j in range(10):  # 最大10回の繰り返し
            # 二次補間により新しい候補ステップ幅を計算
            tau = -nab_fTd * alpha_h / (2 * (f_new - f0 - nab_fTd * alpha_h))
            alpha_zoom = max(0.1, min(tau, 0.9)) * alpha_h  # α_h の0.1〜0.9倍に制限

            # Armijo条件を満たさない、または関数値が悪化している場合
            if (
                obj_f(x_k + alpha_zoom * d_k) > f0 + sig1 * alpha * nab_fTd
                or f_new >= f_old
            ):
                alpha_l = alpha_zoom  # 区間を狭める（左端更新）

            # 新しい点での勾配方向を評価
            nab_fTd_new = nab_f(x_k + alpha_zoom * d_k) @ d_k

            # 強Wolfe条件（勾配が十分に減少）を満たした場合
            if sig2 * np.abs(nab_fTd) >= np.abs(nab_fTd_new):
                return alpha_zoom  # ステップ幅を確定

            # 勾配符号の変化を確認し、探索区間を更新
            if nab_fTd_new * (alpha_h - alpha_l) >= 0:
                alpha_h = alpha_l
            alpha_l, f_old = alpha_zoom, f_new
            f_new = obj_f(x_k + alpha_zoom * d_k)

        # 収束しない場合でも最後のステップ幅を返す
        return alpha_zoom

    # -------------------------------------------------------------
    # メインループ（アルゴリズム2.4）
    # αを段階的に更新しながら Wolfe条件を探索する。
    # -------------------------------------------------------------
    f_new = f_old = obj_f(x_k + alpha * d_k)

    for i in range(10):  # 最大10回試行
        # Armijo条件を満たさない、または関数値が増加している場合
        if f_new > f0 + sig1 * alpha * nab_fTd or (i > 0 and f_new >= f_old):
            alpha = zoom(alpha_old, alpha, x_k, d_k, f_new, f_old)
            return alpha

        # 新しい点での勾配方向
        gd_new = nab_f(x_k + alpha * d_k) @ d_k

        # 強Wolfe条件（勾配が十分に減少）を満たした場合
        if np.abs(nab_fTd) >= sig2 * np.abs(gd_new):
            return alpha

        # 勾配が正に転じた場合 → 最小値を通過したと判断しZoomを使用
        if gd_new >= 0:
            alpha = zoom(alpha_old, alpha, x_k, d_k, f_new, f_old)
            return alpha

        # ここまでで条件を満たさない場合は、ステップ幅を更新して続行
        alpha_old = alpha
        alpha = (alpha + alpha_max) / 2  # 2分法でステップ幅を増加
        f_old = f_new
        f_new = obj_f(x_k + alpha * d_k)
