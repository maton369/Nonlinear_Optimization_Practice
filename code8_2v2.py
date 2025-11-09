import numpy as np

"""
コード8.2 ■ 近接勾配法(バックトラッキング) のコード（点列のログを残すバージョン）
"""


def ProximalGradient_backtrack(
    obj_f,  # f(x): 滑らかな目的関数（勾配がL-リプシッツ連続）
    nab_f,  # ∇f(x): fの勾配を返す関数
    phi,  # φ(x): 非滑らかな凸関数（L1正則化や集合の指示関数など）
    prox,  # prox_{αC}(·): 近接写像（例：soft-threshold, 射影など）
    C,  # 近接写像のスケール（例：φ(x)=λ||x||_1 の λ）
    x_k,  # 初期点
    alpha=1,  # 初期ステップ幅（バックトラッキングの開始値）
    tau=0.5,  # 縮小率（0<tau<1）。alpha ← tau*alpha で段階的に縮小
    sig=0.1,  # Armijo係数 σ（0<sig<1）。十分減少条件の厳しさを決める
    max_iter=10000,  # 最大反復回数
    eps=1.0e-6,  # 収束判定閾値（||α G_k|| ≤ eps）
):
    """
    目的：合成最適化
        最小化  F(x) = f(x) + φ(x)
    を、近接勾配法（バックトラッキング付き）で解く。
    - バックトラッキング：未知のリプシッツ定数に対して、十分減少を満たす α を自動調整。
    - 点列ログ：各反復の x_k を保存して np.array で返す（可視化・解析用）。
    """

    # F_k = f(x_k) + φ(x_k) を初期化（直線探索の基準値）
    F_k = obj_f(x_k) + phi(x_k)

    # 点列ログ（各反復のxを蓄積）。先頭は初期点
    seq_x = [x_k]

    for k in range(max_iter):
        # --- 勾配ステップ（滑らか項 f） ---
        # y_k = x_k - α ∇f(x_k)
        nab_f_k = nab_f(x_k)

        # --- 近接ステップ（非滑らか項 φ） ---
        # T_k = prox_{αC}( y_k )
        T_k = prox(x_k - alpha * nab_f_k, alpha * C)

        # --- 勾配写像（gradient mapping）---
        # G_k = (x_k - T_k) / α
        # これは一階の最適性条件 0 ∈ ∇f(x*) + ∂φ(x*) の達成度を測る量。
        G_k = (x_k - T_k) / alpha

        # 直線探索のために前回の目的値を退避
        F_old = F_k
        # 候補点 T_k における目的値を評価
        F_k = obj_f(T_k) + phi(T_k)

        # --- バックトラッキング（Armijo型十分減少条件）---
        # 条件： F(T_k) ≤ F(x_k) - σ α ||G_k||^2
        # この条件を満たすまで α を tau 倍に縮小して再評価する。
        while F_k > F_old - sig * alpha * np.linalg.norm(G_k) ** 2:
            alpha = alpha * tau
            T_k = prox(x_k - alpha * nab_f_k, alpha * C)
            G_k = (x_k - T_k) / alpha
            F_k = obj_f(T_k) + phi(T_k)

        # --- 点列の更新 ---
        x_k = T_k
        seq_x.append(x_k)

        # --- 収束判定 ---
        # スケール不変な指標として ||α G_k|| ≤ eps を利用（≡ ||G_k|| ≤ eps/α）
        if np.linalg.norm(alpha * G_k) <= eps:
            break

    # 各行が反復ごとの x を表す 2次元配列を返却（プロットや収束可視化に有用）
    return np.array(seq_x)
