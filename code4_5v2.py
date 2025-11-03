"""
コード4.5 ■ 信頼領域法（アルゴリズム4.1：Dogleg法） 点列ログ付き・説明コメント版
本コードは、準ニュートン近似 B_k（BFGSのB公式で更新）を用い、
信頼領域内で二次モデル
    m_k(s) = f(x_k) + g_k^T s + (1/2) s^T B_k s   （g_k = ∇f(x_k)）
を減少させるステップ s_k を Dogleg 法で近似計算する実装である。
各反復で得られた点 x_k を順次 seq_x に記録して返すである。
パラメータの意味：
- Delta_k : 信頼領域半径 Δ_k の初期値
- xi1, xi2 : 受容判定に用いる ρ の閾値（0 < xi1 < xi2 < 1 を推奨）
- eta1, eta2 : Δ_k の縮小・拡大倍率（0 < eta1 < 1 < eta2）
数値上の注意：
- B_k は対称正定を仮定する（BFGS の曲率条件 s^T y > 0 が重要）である。
- ρ の分母（モデル減少量）が非正・極小の場合はステップ拒否側に倒すと安全である。
"""

import numpy as np
from code4_4 import *  # BFGS公式（B公式）: BFGS_B を読み込むである


def dogleg(
    obj_f,
    nab_f,
    x_k,
    *,
    max_iter: int = 1000,
    eps: float = 1.0e-8,
    Delta_k: float = 1.0,
    xi1: float = 0.25,
    xi2: float = 0.75,
    eta1: float = 0.5,
    eta2: float = 2.0
):
    """
    信頼領域法（Dogleg）を用いて最小化を行い、点列ログを返すである。

    Parameters
    ----------
    obj_f : callable
        目的関数 f(x) を返す関数である。
    nab_f : callable
        勾配 ∇f(x) を返す関数である。
    x_k : np.ndarray
        初期点（形状 (n,)）である。
    max_iter : int
        最大反復回数である。
    eps : float
        収束判定用の勾配ノルム閾値である。
    Delta_k : float
        初期信頼領域半径 Δ_0 である。
    xi1, xi2 : float
        ステップ受容のしきい値である（ρ ≥ xi1 で受容、ρ > xi2 で Δ 拡大）。
    eta1, eta2 : float
        信頼領域半径の縮小・拡大倍率である。

    Returns
    -------
    np.ndarray
        反復点列を格納した配列（各行が x_k）である。
    """

    # --- 初期化 -----------------------------------------------------------
    seq_x = [x_k]  # 点列ログを初期化するである
    nab_f_k = nab_f(x_k)  # g_k = ∇f(x_k)
    B_k = np.identity(len(x_k))  # B_0 = I（対称正定の初期近似）
    nab_f_k_old = nab_f_k  # y_k 計算用の前回勾配
    nab_f_k_norm = np.linalg.norm(nab_f_k)  # ||g_k||（収束判定用）

    # --- 反復開始 ---------------------------------------------------------
    for k in range(max_iter):

        # 収束判定：勾配ノルムが十分小さければ終了するである
        if nab_f_k_norm <= eps:
            break

        # ================================================================
        # 信頼領域サブ問題の近似解：Dogleg ステップ s_k を求めるである
        # ================================================================

        # (1) 準ニュートン方向 s_N を計算（B_k s_N = -g_k を solve で解く）
        #     逆行列は明示的に計算せず、線形方程式を解くのが安定である。
        s_N = np.linalg.solve(B_k, -nab_f_k)
        s_N_norm = np.linalg.norm(s_N)

        if s_N_norm <= Delta_k:
            # 準ニュートン方向が信頼領域内 → そのまま採用するである
            s_k = s_N
        else:
            # (2) Cauchy 点 s_C（勾配方向の一次元最適解）を計算するである
            #     s_C = - (||g||^2 / (g^T B g)) * g
            Bnabf = B_k @ nab_f_k
            denom = float(nab_f_k @ Bnabf)
            # 数値ガード：denom ≤ 0 は B が不定/悪条件の兆候である
            if denom <= 0:
                # 最低限の防御として、信頼領域境界上の勾配方向にフォールバックするである
                s_k = -(Delta_k / nab_f_k_norm) * nab_f_k
            else:
                s_C = -(nab_f_k_norm**2 / denom) * nab_f_k
                s_C_norm = np.linalg.norm(s_C)

                if s_C_norm >= Delta_k:
                    # Cauchy 点が領域外 → 境界上の勾配方向に切り詰めるである
                    s_k = -(Delta_k / nab_f_k_norm) * nab_f_k
                else:
                    # (3) s_C と s_N を結ぶ線分 s(τ) = s_C + τ(s_N - s_C) 上で
                    #     境界 ||s(τ)|| = Δ を満たす τ ∈ (0,1] を解く（Dogleg 第2脚）
                    d = s_N - s_C
                    a = float(d @ d)
                    b = float(2.0 * (s_C @ d))
                    c = float(s_C_norm**2 - Delta_k**2)
                    disc = b * b - 4 * a * c
                    disc = max(disc, 0.0)  # 丸め誤差で負になる場合のクリップ
                    tau = (-b + np.sqrt(disc)) / (2.0 * a)  # 0 < τ ≤ 1 を選択
                    s_k = s_C + tau * d

        # ================================================================
        # ステップ受容判定：実減少量とモデル減少量の比 ρ を計算するである
        # ================================================================
        f_now = obj_f(x_k)
        f_new = obj_f(x_k + s_k)
        model_dec = -(nab_f_k @ s_k) - 0.5 * (s_k @ (B_k @ s_k))  # m_k(0) - m_k(s_k)
        if model_dec <= 0:
            rho = -np.inf  # モデルの信頼性が低い → 拒否側に倒すである
        else:
            rho = (f_now - f_new) / model_dec

        if rho >= xi1:
            # --- ステップ受容：点を更新し、曲率情報で B_k を更新するである ---
            x_k = x_k + s_k
            nab_f_k = nab_f(x_k)
            nab_f_k_norm = np.linalg.norm(nab_f_k)

            # 曲率ペア（s_k, y_k）で BFGS(B) 更新（s^T y > 0 で正定性を保ちやすい）
            y_k = nab_f_k - nab_f_k_old
            nab_f_k_old = nab_f_k
            B_k = BFGS_B(B_k, s_k, y_k)

            # 非常に良いステップなら信頼領域を拡大
            if rho > xi2:
                Delta_k = eta2 * Delta_k

            # ログに新しい点を追加するである
            seq_x.append(x_k)

        else:
            # --- ステップ拒否：点は据え置き、信頼領域を縮小するである ---
            Delta_k = eta1 * Delta_k

    # 反復情報の出力（最終反復回数と目的関数値）である
    print("TrustRegion, iter:", k + 1, "f_val:", obj_f(x_k))

    # 点列ログを配列で返すである（各行が1ステップの x_k）
    return np.array(seq_x)
