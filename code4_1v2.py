"""
コード4.1 ■ ニュートン法（点列のログを残す + 直線探索バージョン）
本コードは非線形最適化におけるニュートン法を、Armijo条件に基づくバックトラッキング線探索と
点列ログ（各反復点 x_k の記録）を併用して実装したものであるである。

・更新方向 d_k はヘッセ行列 H_k = ∇^2 f(x_k) と勾配 g_k = ∇f(x_k) に対し
    H_k d_k = -g_k
  を解くことで得る（純粋ニュートン方向）である。

・ステップ幅 α は Armijo 条件
    f(x_k + α d_k) ≤ f(x_k) + σ₁ α ∇f(x_k)^T d_k
  を満たすように line_Armijo で決定するである。

・各反復点を seq_x に順次保存し、最終的に NumPy 配列として返すである。
"""

import numpy as np
from code2_1 import *  # Armijo 条件に基づくバックトラッキング線探索 line_Armijo を読み込む


def Newton(obj_f, nab_f, nab2_f, x_k, max_iter=1000, eps=1.0e-8):
    """
    Armijo線探索付きニュートン法により最小化問題の解を探索し、反復点列を返す関数である。

    Parameters
    ----------
    obj_f : callable
        目的関数 f(x) を返す関数である。
    nab_f : callable
        勾配 ∇f(x)（1階微分ベクトル）を返す関数である。
    nab2_f : callable
        ヘッセ行列 ∇^2 f(x)（2階微分の正方行列）を返す関数である。
    x_k : np.ndarray
        初期点である（形状は (n,) を想定する）。
    max_iter : int, default=1000
        最大反復回数である。
    eps : float, default=1e-8
        収束判定の閾値である（||∇f(x_k)|| ≤ eps なら停止）。

    Returns
    -------
    np.ndarray
        反復ごとの点を格納した配列である。各行が各ステップの x_k を表す。
    """

    # 点列ログを初期化するである（初期点をまず記録）
    seq_x = [x_k]

    for k in range(max_iter):

        # --- (1) ニュートン方向 d_k を計算するである --------------------
        # H_k d_k = -g_k を解く。逆行列は明示的に求めず、solve を用いるのが数値的に安定である。
        H_k = nab2_f(x_k)  # ヘッセ行列（n×n）
        g_k = nab_f(x_k)  # 勾配ベクトル（n,）

        # 数値不安定性（特異・不定）に備えた簡易ガード（任意）である。
        # 失敗時は最急降下方向へフォールバックすることも一法である。
        try:
            d_k = np.linalg.solve(H_k, -g_k)
        except np.linalg.LinAlgError:
            # フォールバック：負の勾配方向（ダンピングを伴う最急降下方向）である。
            d_k = -g_k

        # --- (2) Armijo 条件に基づく線探索でステップ幅 α を決めるである ----
        # 直線探索を導入することで、純粋ニュートンの過大ステップによる発散を緩和できるである。
        alpha = line_Armijo(obj_f, nab_f, x_k, d_k)

        # --- (3) 点を更新し、ログに記録するである ------------------------
        x_k = x_k + alpha * d_k
        seq_x.append(x_k)

        # --- (4) 収束判定：勾配ノルムの小ささで判断するである ----------
        if np.linalg.norm(nab_f(x_k)) <= eps:
            break

    # --- (5) 反復情報を出力するである ------------------------------------
    print("Newton, iter:", k + 1, "f(x):", obj_f(x_k))

    # --- (6) 点列ログを NumPy 配列に変換して返すである --------------------
    return np.array(seq_x)
