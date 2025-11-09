"""
コード8.3 ■ FISTA(アルゴリズム8.4) のコード（点列のログを残すバージョン：説明コメント付き）
"""

import numpy as np


def FISTA(obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps=1.0e-6):
    """
    FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）
    合成最適化
        minimize_x  F(x) = f(x) + φ(x)
    を Nesterov 型の加速を用いて解く。ここでは各反復の点 x_k を保存して返す。

    引数:
      obj_f : f(x) を返す関数（監視用：ここでは値を返すだけ）
      nab_f : ∇f(x) を返す関数（L-リプシッツ連続と仮定）
      phi   : φ(x) を返す関数（監視用：目的値表示などに使用可能）
      prox  : 近接写像 prox_{αC}(·)（例：L1 なら soft-threshold、射影など）
      C     : 近接写像のスケール（例：φ(x)=λ||x||_1 の λ）
      x_k   : 初期点
      alpha : ステップ幅（目安は α ≤ 1/L。L は ∇f のリプシッツ定数）
      max_iter : 反復上限
      eps      : 収束判定閾値（||x_k - x_{k-1}|| ≤ eps）

    戻り値:
      np.ndarray 形状 (反復回数+1, n)。各行が反復で得られた点（可視化・解析に有用）
    """

    seq_x = [x_k]  # 反復で得られる主系列 x_k を記録（先頭は初期点）
    y_k = x_k  # 加速点の初期化（通常、初期は主系列と同じにする）
    tau_k = 1.0  # 加速パラメータ t_k（FISTA 標準初期値は 1）

    for k in range(max_iter):
        # --- 退避と勾配計算 ---
        # x_old はモメンタム項の計算に用いるため保持する
        x_old = x_k
        # FISTA は勾配評価を加速点 y_k で行う（Nesterov 加速の要点）
        nab_f_k = nab_f(y_k)

        # --- 近接勾配ステップ（主系列の更新）---
        # x_k = prox_{αC}( y_k - α ∇f(y_k) )
        # ここで prox は φ に応じた近接写像（例：L1→soft、制約→射影）
        x_k = prox(y_k - alpha * nab_f_k, alpha * C)

        # --- 加速パラメータの更新 ---
        # tau_{k+1} = (1 + sqrt(1 + 4 tau_k^2)) / 2
        # これにより目的値の収束率 O(1/k^2) を達成（PGM の O(1/k) より速い）
        tau_new = (1.0 + np.sqrt(1.0 + 4.0 * tau_k**2)) / 2.0

        # --- 加速点の更新（Nesterov モメンタム）---
        # y_{k+1} = x_k + ((tau_k - 1)/tau_{k+1}) * (x_k - x_{k-1})
        # 「主系列 x の差分」を利用しつつ、次の勾配評価点を前方に進める
        y_k = x_k + ((tau_k - 1.0) / tau_new) * (x_k - x_old)

        # --- ログの保存 ---
        seq_x.append(x_k)

        # --- 収束判定 ---
        # 主系列の変化量が十分小さくなったら停止（他に ||G|| や目的値差でも可）
        if np.linalg.norm(x_k - x_old) <= eps:
            break

        # --- 次反復に向けてパラメータ更新 ---
        tau_k = tau_new

    # 反復で得られた点列（各行が x_k）を返す
    return np.array(seq_x)
