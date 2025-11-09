# -----------------------------------------------------------------------------
# コード8.6 ■ ニュートン型近接勾配法(アルゴリズム8.5) のコード（説明コメント付き）
# -----------------------------------------------------------------------------
# 目的：
#   合成最適化問題
#       minimize_x  F(x) = f(x) + φ(x)
#   を「滑らか項 f にはニュートン近似（2次モデル）」「非滑らか項 φ には近接写像」
#   という役割分担で解く手法（Proximal Newton Method）の基本実装である。
#
# 基本アイデア：
#   現在点 x_k で f を 2次関数に近似し、次の部分問題（サブ問題）を解く：
#       minimize_x   Q_k(x) + φ(x)
#     ここで Q_k(x) = f(x_k) + ∇f(x_k)^T (x - x_k) + 1/2 (x - x_k)^T B_k (x - x_k)
#           B_k ≈ ∇^2 f(x_k)（ヘッセ行列またはその近似）
#   この合成最小化は一般に closed-form にならないため、
#   「近接勾配法（バックトラッキング付き）」で数値的に解く（1ステップで厳密に解く必要はない）。
#   得られた解 x_k_plus と元の点 x_k の差 d_k = x_k_plus - x_k を「ニュー・トン方向の近接版」とみなし、
#   Armijo 型の直線探索でステップ幅 α を選んで更新する。
#
# 収束の直感：
#   強凸かつ B_k が正定のとき、近傍では二次収束／準二次収束が期待でき、
#   φ によるスパース化や制約の効果は prox 側で自然に扱われる。
#
# 注意：
#   - B_k は対称正定が望ましい（Hess(x_k) が不定の場合は正則化を検討：B_k ← B_k + δI）。
#   - 直線探索の十分減少条件には「合成の一次減少量」
#         Δ_k = ∇f(x_k)^T d_k + φ(x_k_plus) - φ(x_k)
#     を用いる（Moreau 包絡の観点で妥当な選択）。
#   - サブ問題ソルバとして ProximalGradient_backtrack を再利用している
#     （Q_k + φ の最小化＝重み付き近接写像の近似計算）。
# -----------------------------------------------------------------------------

import numpy as np
from code8_2 import *  # 近接勾配法(バックトラッキング) ProximalGradient_backtrack を読み込み


def Proximal_Newton(
    obj_f,  # f(x) : 滑らかな目的関数
    nab_f,  # ∇f(x): f の勾配
    phi,  # φ(x) : 非滑らかな凸関数（L1, 指示関数など）
    prox,  # prox_{·}(·): φ に対応する近接写像（例：soft-threshold, 射影）
    C,  # 近接スケール（例：φ(x)=λ||x||_1 なら C=λ）
    x_k,  # 初期点
    Hess,  # ∇^2 f(x) を返す関数（もしくはその近似ヘッセ）
    tau=0.5,  # 直線探索の縮小率（0<tau<1）
    sig=0.1,  # Armijo 係数 σ（0<sig<1）
    max_iter=1000,
    eps=1.0e-6,
):
    # 初期のヘッセ行列 B_k と勾配 ∇f(x_k)、目的関数値 F(x_k)=f(x_k)+φ(x_k) を評価
    B_k, nab_f_k = Hess(x_k), nab_f(x_k)
    F_k = obj_f(x_k) + phi(x_k)

    for k in range(max_iter):
        # ---------------- サブ問題（2次モデル＋非滑らか項）の定義 ----------------
        # QP(x) = f(x_k) + ∇f(x_k)^T (x - x_k) + 1/2 (x - x_k)^T B_k (x - x_k)
        # nab_QP(x) = ∇f(x_k) + B_k (x - x_k)
        # これを φ(x) と合成した最小化 QP(x) + φ(x) を PGM-Backtracking で解く。
        QP = (
            lambda x: obj_f(x_k)
            + nab_f_k.T @ (x - x_k)
            + 0.5 * (B_k @ (x - x_k)).T @ (x - x_k)
        )
        nab_QP = lambda x: nab_f_k + B_k @ (x - x_k)

        # ---------------- サブ問題の解法（近接勾配法 with backtracking） ----------------
        # ここで得られる x_k_plus は厳密解でなくてもよい（1ステップでも実務上は有効）。
        # 実装上は ProximalGradient_backtrack が内部で Armijo 条件を満たすまで α を調整。
        x_k_plus = ProximalGradient_backtrack(QP, nab_QP, phi, prox, C, x_k)

        # 近接ニュートン方向（擬似ニュートン方向）：d_k = x_k_plus - x_k
        d_k = x_k_plus - x_k

        # 直線探索の初期化（α=1 から開始）
        alpha, F_old = 1.0, F_k

        # 合成の一次減少量 Δ_k（Armijo 条件の右辺に入る「期待減少」）
        #   Δ_k = ∇f(x_k)^T d_k + φ(x_k_plus) - φ(x_k)
        Delta_k = nab_f_k.T @ d_k + phi(x_k_plus) - phi(x_k)

        # 候補点（α=1）の目的関数値を評価
        F_k = obj_f(x_k_plus) + phi(x_k_plus)

        # ---------------- Armijo 型直線探索（合成十分減少）----------------
        # 条件： F(x_k + α d_k) ≤ F(x_k) + σ α Δ_k
        #       （Δ_k ≤ 0 なら右辺は減少方向を示す）
        alpha = 1.0
        while F_k > F_old + alpha * sig * Delta_k:
            alpha = alpha * tau  # ステップ幅を縮小
            F_k = obj_f(x_k + alpha * d_k) + phi(x_k + alpha * d_k)

        # ---------------- 更新と停止判定 ----------------
        x_k = x_k + alpha * d_k  # 新しい点へ更新
        print(f"Proximal_Newton:反復回数{k+1:d}, 最適値{F_k:.5e}")

        # 方向ノルムで停止（他に ||G_k|| や KKT 残差でも良い）
        if np.linalg.norm(d_k) <= eps:
            break

        # 次反復に向けて B_k と ∇f を更新（必要なら B_k に正則化を追加）
        B_k, nab_f_k = Hess(x_k), nab_f(x_k)

    return x_k
