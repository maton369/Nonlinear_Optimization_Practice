"""
コード4.5 ■ 信頼領域法（アルゴリズム4.1：Dogleg法） のコード（説明コメント付き）
本実装は、BFGS の B 公式でヘッセ近似 B_k を更新しつつ、信頼領域内のサブ問題を
Dogleg（ドッグレッグ）で近似的に解く信頼領域法である。

【概要】
- 信頼領域半径 Δ_k の内側で二次モデル
      m_k(s) = f(x_k) + g_k^T s + 1/2 s^T B_k s
  を十分減少させるステップ s_k を求めるである（g_k = ∇f(x_k)）。
- Dogleg は「Cauchy点（勾配降下方向）」 s_C と「（準）Newton 方向」 s_N を結ぶ折れ線上で
  最適な点を選ぶ近似解法である。
- 実際の減少量とモデル減少量の比
      ρ_k = (f(x_k) - f(x_k + s_k)) / (m_k(0) - m_k(s_k))
  に基づき、受容/拒否および信頼領域半径 Δ_k の拡大/縮小を決めるである。

【ハイパーパラメータ】
- Delta_k : 初期信頼領域半径
- xi1, xi2 : ステップ受容のしきい値（0 < xi1 < xi2 < 1 を想定）
             ρ_k ≥ xi1 なら受容、ρ_k > xi2 なら「非常に良い」とみなし Δ_k を拡大
- eta1, eta2 : Δ_k の縮小・拡大倍率（0 < eta1 < 1 < eta2）

【注意点】
- B_k は対称正定を想定する（BFGS(B) の曲率条件 s^T y > 0 がこれを支える）。
- s_N = B_k^{-1} (-g_k) は線形方程式を solve で解く（逆行列は計算しない）である。
"""

import numpy as np
from code4_4 import *  # BFGS公式（B 公式）を読み込むである：BFGS_B


def dogleg(
    obj_f,
    nab_f,
    x_k,
    *,
    max_iter: int = 1000,
    eps: float = 1e-8,
    Delta_k: float = 1.0,  # 初期信頼領域半径 Δ_0
    xi1: float = 0.25,  # 受容下限しきい値
    xi2: float = 0.75,  # 良好受容しきい値（これを超えたら Δ を拡大）
    eta1: float = 0.5,  # Δ 縮小倍率
    eta2: float = 2.0  # Δ 拡大倍率
):
    """
    信頼領域法（Dogleg）により最小化問題の解を探索するである。

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
        収束判定の閾値（||∇f(x_k)|| ≤ eps で停止）である。
    Delta_k : float
        初期信頼領域半径である。
    xi1, xi2 : float
        ステップ受容判定のしきい値である（0 < xi1 < xi2 < 1 を推奨）。
    eta1, eta2 : float
        信頼領域半径 Δ_k の縮小・拡大倍率である。

    Returns
    -------
    np.ndarray
        最終反復点 x_k である。
    """

    # --- 初期化：勾配とヘッセ近似の用意 -----------------------------------
    nab_f_k = nab_f(x_k)  # g_k = ∇f(x_k)
    B_k = np.identity(len(x_k))  # B_0 = I（対称正定の初期化）
    nab_f_k_old = nab_f_k  # y_k 計算用に前回勾配を保持
    nab_f_k_norm = np.linalg.norm(nab_f_k)  # ||g_k||（収束判定に用いる）

    for k in range(max_iter):

        # --- 収束判定：勾配ノルムが十分小さければ停止 --------------------
        if nab_f_k_norm <= eps:
            break

        # === 信頼領域サブ問題の近似解（Dogleg）を求める ===================

        # (1) 準ニュートン方向 s_N = argmin_s m_k(s)（B_k s = -g_k を解く）
        #     これは二次モデルの臨界点であり、信頼領域内なら最適候補である。
        s_N = np.linalg.solve(B_k, -nab_f_k)
        s_N_norm = np.linalg.norm(s_N)

        if s_N_norm <= Delta_k:
            # (1-a) 準ニュートン方向が信頼領域内 → そのまま採用
            s_k = s_N

        else:
            # (2) Cauchy点 s_C：勾配方向に沿った一次元最適化の解（モデル最良の勾配方向ステップ）
            #     s_C = - (||g||^2 / (g^T B g)) * g
            Bnabf = B_k @ nab_f_k
            s_C = -(nab_f_k_norm**2 / (nab_f_k @ Bnabf)) * nab_f_k
            s_C_norm = np.linalg.norm(s_C)

            if s_C_norm >= Delta_k:
                # (2-a) Cauchy点が信頼領域外 → 境界上の勾配方向スケールを採用
                #       s = - (Δ / ||g||) g
                s_k = -(Delta_k / nab_f_k_norm) * nab_f_k

            else:
                # (3) s_C と s_N を結ぶ線分上で、信頼領域境界と交わる点を選ぶ（Dogleg の第2脚）
                #     すなわち s(τ) = s_C + τ (s_N - s_C), τ ∈ (0, 1] が ||s(τ)|| = Δ を満たす τ を求める。
                #     ||s_C + τ d||^2 = Δ^2  の 2次方程式を解く（d = s_N - s_C）。
                d = s_N - s_C
                sNsC = s_N @ s_C
                sNsC_norm2 = (np.linalg.norm(d)) ** 2  # = ||s_N - s_C||^2

                # 係数展開：||s_C + τ d||^2 = ||s_C||^2 + 2τ s_C^T d + τ^2 ||d||^2 = Δ^2
                # → τ^2 ||d||^2 + 2τ (s_C^T d) + (||s_C||^2 - Δ^2) = 0
                a = sNsC_norm2
                b = 2.0 * (s_C @ d)
                c = s_C_norm**2 - Delta_k**2
                disc = b * b - 4 * a * c
                # 浮動小数点により負になる可能性に備えてクリップ
                disc = max(disc, 0.0)
                tau = (-b + np.sqrt(disc)) / (2.0 * a)  # 0 < τ ≤ 1 を選ぶ
                s_k = s_C + tau * d

        # === ステップ受容判定と信頼領域半径の調整 ==========================

        # 実際の減少量 / モデル減少量：
        #   ρ = (f(x_k) - f(x_k + s_k)) / ( - g_k^T s_k - 0.5 s_k^T B_k s_k )
        # 分母は m_k(0) - m_k(s_k) の解析式である。
        f_now = obj_f(x_k)
        f_new = obj_f(x_k + s_k)
        model_dec = -(nab_f_k @ s_k) - 0.5 * (s_k @ (B_k @ s_k))
        # 数値安全のため、モデル減少が極小/負になる場合のガード
        if model_dec <= 0:
            rho = -np.inf  # 受容しない方向へ
        else:
            rho = (f_now - f_new) / model_dec

        if rho >= xi1:
            # --- ステップ受容：点を更新し、BFGS(B) で曲率を反映 ---------
            x_k = x_k + s_k
            nab_f_k = nab_f(x_k)
            nab_f_k_norm = np.linalg.norm(nab_f_k)

            # 曲率ペア s_k, y_k を計算して B_k を更新する（BFGS は正定性を保ちやすい）
            y_k = nab_f_k - nab_f_k_old
            nab_f_k_old = nab_f_k
            B_k = BFGS_B(B_k, s_k, y_k)

            # 非常に良いステップであれば信頼領域を拡大
            if rho > xi2:
                Delta_k = eta2 * Delta_k

        else:
            # --- ステップ拒否：点は据え置き、信頼領域を縮小 --------------
            Delta_k = eta1 * Delta_k

    print("TrustRegion, iter:", k + 1, "f_val:", obj_f(x_k))
    return x_k
