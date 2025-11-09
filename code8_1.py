# 近接勾配法（Proximal Gradient Method; PGM）の定数ステップ幅版である。
# 目的：合成最適化
#     minimize_x  F(x) := f(x) + φ(x)
# ただし f は滑らか（L-リプシッツ連続勾配）、φ は一般に非滑らか＋凸（例：L1正則化、指示関数）を想定する。
# 本実装は一定のステップ幅 α を用い、近接写像 prox により非滑らか項を処理するである。
#
# 反復式：
#   T_k = prox_{α C}( x_k - α ∇f(x_k) )
#   x_{k+1} = T_k
# ここで C は φ に付随する「近接のスケール／ペナルティの重み」を表す（例：φ(x)=λ||x||_1 なら C=λ）。
#
# 最適性計測（勾配写像：gradient mapping）：
#   G_k = (x_k - T_k)/α
#   凸合成問題に対し、||G_k|| → 0 は最適性条件（0 ∈ ∇f(x*) + ∂φ(x*)）の数値的指標として用いられる。
#
# 収束要件の目安：
#  - f の勾配が L-リプシッツ連続のとき、0 < α < 2/L で単調減少性が期待できる（典型には α ≤ 1/L）。
#  - φ が強凸/強単調写像なら加速や線形収束も検討可能である（本関数は定数αの基本形）。
#
# 引数：
#   obj_f : f(x) を返す関数
#   nab_f : ∇f(x) を返す関数
#   phi   : φ(x) を返す関数（監視用；目的値表示に利用）
#   prox  : 近接写像 prox_{α C}(·) を返す関数（例：soft-threshold、射影など）
#   C     : 近接のスケール（例：λ）。prox 側の設計により無視されることもある
#   x_k   : 初期点
#   alpha : 定数ステップ幅 α
#   max_iter : 反復上限
#   eps      : 収束判定閾値（||α G_k|| ≤ eps）
#
# 出力：
#   x_k : 反復後の推定解
#
# 備考（実務的ポイント）：
#  - 射影付き最小化（φ が集合の指示関数）では prox は集合への射影になる（例：ボックス制約）。
#  - L1 正則化（φ(x)=λ||x||_1）では prox は成分毎のソフトしきい値化 soft(x, αλ) になる。
#  - 収束が遅い場合、バックトラッキングで α を調整する Armijo 型 PGM や FISTA（加速）を検討する。
import numpy as np


def ProximalGradient_const(
    obj_f, nab_f, phi, prox, C, x_k, alpha, max_iter=10000, eps=1.0e-6
):
    for k in range(max_iter):
        # 勾配ステップ：滑らか項 f の勾配で前進（x_k - α∇f(x_k)）
        # 近接ステップ：非滑らか項 φ に対する近接写像で正則化・拘束を反映
        T_k = prox(x_k - alpha * nab_f(x_k), alpha * C)  # 近接勾配法の基本1ステップ

        # 勾配写像（gradient mapping）：最適性の指標
        # G_k = 0 に近いほど KKT/最適性条件 0 ∈ ∇f(x*) + ∂φ(x*) に近づいていることを示す
        G_k = (x_k - T_k) / alpha

        # 次の反復点に更新
        x_k = T_k

        # 収束判定：||α G_k|| ≤ eps（スケール不変な基準として ||G_k|| ≤ eps/α でも同等）
        if np.linalg.norm(alpha * G_k) <= eps:
            break

    # 反復回数と目的値（f+φ）を表示（解析・学習用）
    print(f"PGM with const.:反復回数{k+1:d}, 最適値{obj_f(x_k) + phi(x_k):.5e}")
    return x_k
