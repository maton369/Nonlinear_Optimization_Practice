# 近接勾配法（Proximal Gradient Method; PGM）定数ステップ幅版の
# 「点列ログを返す」バージョンである。
# 目的：合成最適化
#     minimize_x  F(x) := f(x) + φ(x)
# ただし f は滑らか（∇f が L-リプシッツ連続）、φ は一般に非滑らか＋凸（例：L1正則化、制約の指示関数）を想定する。
#
# 反復式（定数ステップ幅 α）：
#   1) 勾配ステップ：  y_k = x_k - α ∇f(x_k)
#   2) 近接ステップ：  T_k = prox_{α C}(y_k)
#   3) 更新：            x_{k+1} = T_k
# ここで C は φ に付随する重み（例：φ(x)=λ||x||_1 なら C=λ）。prox 側の実装により無視される場合もある。
#
# 勾配写像（gradient mapping）：
#   G_k = (x_k - T_k) / α
#   凸合成問題に対し、||G_k|| → 0 は一階の最適性条件 0 ∈ ∇f(x*) + ∂φ(x*) を満たす指標となる。
#
# 収束の目安：
#   ∇f が L-リプシッツ連続なら、0 < α < 2/L（典型には α ≤ 1/L）で単調減少・収束が期待できる。
#   収束が遅い場合はバックトラッキングや加速（FISTA）を検討する。
#
# 入出力：
#   obj_f : f(x) を返す関数（監視用。ここでは返却値には使わない）
#   nab_f : ∇f(x) を返す関数
#   phi   : φ(x) を返す関数（監視用。ここでは返却値には使わない）
#   prox  : 近接写像 prox_{α C}(·) を返す関数（例：soft-threshold、集合への射影）
#   C     : 近接のスケール（例：λ）
#   x_k   : 初期点
#   alpha : 定数ステップ幅 α
#   max_iter : 最大反復回数
#   eps      : 収束判定閾値（||α G_k|| ≤ eps）
# 戻り値：
#   np.array(seq_x) : 反復で得られた点列（各行が x_k）を返す。可視化や収束解析に用いる。
import numpy as np


def ProximalGradient_const(
    obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps=1.0e-6
):
    seq_x = [x_k]  # 反復で得られる点列を保持（可視化・解析用）
    for k in range(max_iter):
        # 1) 勾配ステップ：滑らか項 f に対して前進（勾配降下）
        #    y_k = x_k - α ∇f(x_k)
        # 2) 近接ステップ：非滑らか項 φ に対する近接写像で正則化/制約効果を反映
        #    T_k = prox_{α C}(y_k)
        T_k = prox(x_k - alpha * nab_f(x_k), alpha * C)  # 近接勾配法の基本1ステップ

        # 勾配写像（gradient mapping）：最適性の尺度
        #   G_k = (x_k - T_k)/α
        #   ||G_k|| が小さいほど KKT/一階条件 0 ∈ ∇f(x*) + ∂φ(x*) に近い
        G_k = (x_k - T_k) / alpha

        # 3) 更新：次の点へ
        x_k = T_k
        seq_x.append(x_k)

        # 収束判定：スケールされた基準 ||α G_k|| ≤ eps
        #   （等価に ||G_k|| ≤ eps/α としてもよい）
        if np.linalg.norm(alpha * G_k) <= eps:
            break

    return np.array(seq_x)
