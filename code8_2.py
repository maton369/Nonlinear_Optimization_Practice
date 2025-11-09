# 近接勾配法（Proximal Gradient Method; PGM）
# バックトラッキング付きステップ幅選択（Armijo型）バージョンである。
#
# 目的：合成最適化
#   minimize_x  F(x) := f(x) + φ(x)
# ただし f は滑らか（∇f が L-リプシッツ連続）、φ は一般に凸・非滑らか（L1正則化や集合の指示関数など）を想定する。
#
# 反復の基本形：
#   y_k = x_k - α ∇f(x_k)            （勾配ステップ）
#   T_k = prox_{α C}(y_k)             （近接ステップ）
#   x_{k+1} = T_k
# ここで C は φ に付随するスケール（例：φ(x)=λ||x||_1 なら C=λ）。
#
# バックトラッキング（直線探索）の目的：
#   ・未知の L（∇f のリプシッツ定数）に対して適切な α を自動調整
#   ・目的関数の十分減少を保証し、発散や過大ステップを防ぐ
#
# 使用する減少条件（Armijo型；Moreau包絡の評価として知られる）：
#   F(T_k) ≤ F(x_k) - σ α ||G_k||^2,
#   ここで G_k := (x_k - T_k)/α は「勾配写像（gradient mapping）」であり、最適性の尺度となる。
#
# 収束判定：
#   ||α G_k|| ≤ ε  （≡ ||G_k|| ≤ ε/α）
#   凸合成問題に対して、||G_k|| → 0 は一階条件 0 ∈ ∇f(x*) + ∂φ(x*) の数値的達成を示す。
#
# 引数：
#   obj_f : f(x) を返す関数
#   nab_f : ∇f(x) を返す関数
#   phi   : φ(x) を返す関数
#   prox  : 近接写像 prox_{α C}(·) を返す関数（例：ソフトしきい値化、集合への射影）
#   C     : 近接のスケール（例：λ）
#   x_k   : 初期点
#   alpha : 初期ステップ幅（バックトラッキングの開始値）
#   tau   : 縮小率（0 < tau < 1）。例：0.5
#   sig   : Armijo係数 σ（0 < sig < 1）。例：0.1
#   max_iter : 最大反復回数
#   eps      : 収束判定閾値
#
# 出力：
#   x_k : 反復終了時の推定解
import numpy as np


def ProximalGradient_backtrack(
    obj_f,
    nab_f,
    phi,
    prox,
    C,
    x_k,
    alpha=1,
    tau=0.5,
    sig=0.1,
    max_iter=10000,
    eps=1.0e-6,
):
    # 現在点での目的値 F(x_k) = f(x_k) + φ(x_k) を初期化
    F_k = obj_f(x_k) + phi(x_k)

    for k in range(max_iter):
        # --- 勾配ステップ（滑らか項 f） ---
        nab_f_k = nab_f(x_k)  # ∇f(x_k) を評価

        # --- 近接ステップ（非滑らか項 φ） ---
        # T_k = prox_{αC}( x_k - α ∇f(x_k) )
        T_k = prox(x_k - alpha * nab_f_k, alpha * C)

        # --- 勾配写像（gradient mapping） ---
        # G_k = (x_k - T_k) / α
        # ||G_k|| が小さいほど最適性条件に近い
        G_k = (x_k - T_k) / alpha

        # 直線探索の基準となる「前回の目的値」を保存
        F_old = F_k
        # 候補点 T_k における目的値を評価
        F_k = obj_f(T_k) + phi(T_k)

        # --- バックトラッキング（Armijo型十分減少条件）---
        # F(T_k) ≤ F(x_k) - σ α ||G_k||^2 を満たすまで α を縮小
        while F_k > F_old - sig * alpha * np.linalg.norm(G_k) ** 2:
            alpha = alpha * tau  # ステップ幅を縮小（0<tau<1）
            T_k = prox(x_k - alpha * nab_f_k, alpha * C)  # 再計算
            G_k = (x_k - T_k) / alpha
            F_k = obj_f(T_k) + phi(T_k)

        # --- 点列の更新 ---
        x_k = T_k

        # --- 収束判定 ---
        # スケール不変な基準として ||α G_k|| ≤ eps を用いる
        if np.linalg.norm(alpha * G_k) <= eps:
            break

    # 進捗ログ（学習・検証用）
    print(f"PGM with backtracking.:反復回数{k+1:d}, 最適値{F_k:.5e}")
    return x_k
