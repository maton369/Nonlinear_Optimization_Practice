# -----------------------------------------------------------------------------
# コード8.8 ■ 一般化Lassoに特化したADMM(アルゴリズム8.6) のコード（説明コメント付き）
# -----------------------------------------------------------------------------
# 問題設定（Generalized Lasso）:
#   minimize_x   (1/2) ||A x - b||_2^2  +  C ||D x||_1
# ここで、A∈R^{m×n}, b∈R^m, D∈R^{ℓ×n}, C>0。
# 典型例：D=I（LASSO）、一次/二次差分行列（全変動正則化, fused lasso など）
#
# ADMMによる分割：
#   introduce  y = D x
#   minimize_{x,y}  (1/2)||A x - b||^2 + C ||y||_1
#   s.t.  Dx - y = 0
#
# 拡張ラグランジュ関数（スケーリング形）:
#   L_ρ(x,y,μ) = (1/2)||A x - b||^2 + C||y||_1 + μ^T(Dx - y) + (ρ/2)||Dx - y||^2
#
# 反復（k=0,1,2,...）：
#   1) x更新： (A^T A + ρ D^T D) x = A^T b + ρ D^T (y - μ/ρ)
#   2) y更新： y = prox_{(C/ρ)||·||_1}(Dx + μ/ρ)  ← ソフト閾値
#   3) 乗数更新： μ ← μ + ρ (Dx - y)
#   4) 停止判定： primal residual r = Dx - y, dual residual s = ρ D^T (y - y_old)
#                 ||r|| ≤ ε_pri, ||s|| ≤ ε_dual
#   5) ρの自動調整：||r|| と ||s|| のバランスが悪いときに ρ を上げ下げ（He et al., Boyd et al.）
#
# 入出力：
#   A,b,D,C         : 問題データ
#   x_k,y_k         : 初期値（ゼロや最小二乗解など）
#   prox_y(v,thr)   : l1用prox（soft-threshold）。thr = C/ρ を渡す
#   rho_k           : ρの初期値
#   eta,zeta1,zeta2 : ρ調整の閾値と倍率（標準的に eta∈[5,10], zeta1=zeta2≈2）
#   eps3,eps4       : 停止閾値（Boyd本に準拠：絶対誤差と相対誤差の合成）
#   戻り値          : (x, y, μ)
#
# メモ：
#  - 行列が大きい場合、(A^T A + ρ D^T D) への直接solveは高コスト。
#    反復法（共役勾配）や行列分解の再利用（factorization caching）が有効。
#  - 実装上の安定性のため、||D^T μ|| の計算は行列積（@）で行うこと（* だと要素積になり誤り）。
# -----------------------------------------------------------------------------
import numpy as np


def generalized_lasso_ADMM(
    A,
    b,
    D,
    C,
    x_k,  # x の初期値（R^n）
    y_k,  # y の初期値（R^ℓ）
    prox_y,  # y更新に用いるprox（例：soft-thresholding）
    rho_k=10,  # ρ の初期値
    max_iter=500,
    eta=10,  # ρ調整の閾値（||r|| > eta*||s|| など）
    zeta1=2,
    zeta2=2,  # ρ の拡大/縮小倍率
    eps3=1.0e-3,  # 絶対誤差項
    eps4=1.0e-4,  # 相対誤差項
):
    # 前計算：A^T A, A^T b, D^T D
    AtA, Atb, DtD = A.T @ A, A.T @ b, D.T @ D

    # 停止判定の閾値（BoydのADMM本に準拠）：
    #   ε_pri = √ℓ * eps3 + eps4 * max(||Dx||, ||y||)
    #   ε_dual = √n * eps3 + eps4 * ||D^T μ||
    sqrt_l_eps = eps3 * np.sqrt(len(y_k))
    sqrt_n_eps = eps3 * np.sqrt(len(x_k))

    # スケーリング乗数 μ（未スケールλに相当）
    mu_k = np.zeros_like(y_k)

    for k in range(max_iter):
        # --- x-update: (A^T A + ρ D^T D) x = A^T b + ρ D^T (y - μ/ρ) ---
        # ここで係数行列は対称正定（ρ>0, Dがフルランクでなくても通常正定化されやすい）。
        # 大規模ではCGや事前分解の再利用を推奨。
        x_k = np.linalg.solve(
            AtA + rho_k * DtD, Atb + rho_k * D.T @ (y_k - mu_k / rho_k)
        )

        # --- y-update: soft-threshold（prox_{(C/ρ)||·||_1}） ---
        Dx = D @ x_k
        y_k_old = np.copy(y_k)
        y_k = prox_y(Dx + mu_k / rho_k, C / rho_k)

        # --- 残差の計算 ---
        r_k = Dx - y_k  # primal residual
        s_k = rho_k * D.T @ (y_k - y_k_old)  # dual residual（D^T μ の変化に対応）

        norm_r, norm_s = np.linalg.norm(r_k), np.linalg.norm(s_k)

        # --- 乗数更新（scaled form） ---
        mu_k = mu_k + rho_k * r_k

        # --- 停止判定の閾値（絶対+相対）---
        eqs1 = sqrt_l_eps + eps4 * max(np.linalg.norm(Dx), np.linalg.norm(y_k))  # ε_pri
        # 注意：D.T*mu_k は要素積になってしまうため、必ず D.T @ mu_k（行列積）を用いる
        eqs2 = sqrt_n_eps + eps4 * np.linalg.norm(D.T @ mu_k)  # ε_dual

        if norm_r <= eqs1 and norm_s <= eqs2:
            break

        # --- ρ の自動調整（residual balancing）---
        if norm_r > eta * norm_s:
            rho_k *= zeta1  # primal が大きい → ρ を増やして拘束を強める
        elif norm_s > eta * norm_r:
            rho_k /= zeta2  # dual が大きい → ρ を減らして動きを許す

    # 目的関数値（(1/2)||Ax-b||^2 + C||Dx||_1）と進捗を簡易表示
    obj_val = 0.5 * np.sum((A @ x_k - b) ** 2) + C * np.linalg.norm(Dx, 1)
    print(
        f"ADMM: 反復回数{k+1:d}, 目的関数値{obj_val:.3e}, "
        f"||r_k||={norm_r:.3e}, ||s_k||={norm_s:.3e}, rho_k={rho_k:.3e}"
    )

    return x_k, y_k, mu_k
