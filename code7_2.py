# 拡張ラグランジュ法（Augmented Lagrangian Method；アルゴリズム7.3）の実装である。
# 目的：等式制約付き最適化
#     minimize   f(x)
#     subject to h(x) = 0
# を、拡張ラグランジュ関数
#     L_ρ(x, μ) = f(x) + μ^T h(x) + ρ * ‖h(x)‖_2^2
# を段階的に最小化しつつラグランジュ乗数 μ を更新することで解く手法である。
# 典型的な定義では二次項が (ρ/2)‖h‖^2 であるが、本実装は係数を ρ としており、
# これは ρ のスケーリングが異なるだけで実質同等である（パラメータ調整で吸収可能）である。
#
# アルゴリズム骨子：
#   1) 初期 (x_0, μ_0, ρ_0) を与える。
#   2) 内側ソルバで x_{k+1} = argmin_x [ f(x) + μ_k^T h(x) + ρ_k ‖h(x)‖^2 ] を解く。
#   3) 収束判定：‖h(x_{k+1})‖ が十分小さければ停止。
#   4) 乗数更新： μ_{k+1} = μ_k + ρ_k h(x_{k+1})
#   5) ペナルティ係数更新： ρ_{k+1} = c · ρ_k（c > 1）として 2) へ戻る。
#
# 実装上の注意：
#   - eq_h(x) は R^n → R^ℓ のベクトル関数を想定するである。
#   - 乗数ベクトル μ_k の長さは h(x) と一致させる必要があるである。
#   - 内側最小化は SciPy optimize.minimize に委譲する。method を指定（例 BFGS）すると挙動が安定しやすいである。
#   - ρ の増加は内側問題の悪条件化を招く可能性があるため、増加率や最大反復は問題依存で調整するである。
#   - 収束判定は ‖h(x_k)‖ に加え、x の変化量や L_ρ の減少量を併用すると堅牢である。

import numpy as np
from scipy import optimize
from typing import Callable


def ALagrangian_method(
    obj_f: Callable[[np.ndarray], float],  # 目的関数 f(x) を返す関数である
    eq_h: Callable[
        [np.ndarray], np.ndarray
    ],  # 等式制約ベクトル h(x)（形状 (ℓ,)）を返す関数である
    x_k: np.ndarray,  # 初期点である（形状 (n,)）
    mu_k: np.ndarray,  # 初期ラグランジュ乗数（形状 (ℓ,)）
    rho: float = 1.0,  # 初期ペナルティ係数 ρ である
    max_iter: int = 10,  # 外側反復の最大回数である
    eps: float = 1e-8,  # 収束判定閾値（‖h(x)‖^2）である
) -> np.ndarray:
    """
    拡張ラグランジュ法で等式制約付き最適化を解くである。
    返り値は最終反復の近似解 x_k である。
    """

    # 乗数ベクトルの形状チェックである（開発時のバグ検知用）
    h0 = eq_h(x_k)
    assert mu_k.shape == h0.shape, "mu_k の形状は h(x) と一致させるである"

    # ペナルティ項 P(x) = ‖h(x)‖^2 を定義するである
    def pen_val(x: np.ndarray) -> float:
        h_vals = eq_h(x)
        return float(np.sum(h_vals**2))

    for k in range(max_iter):
        # 内側最小化問題の目的関数 φ(x) = f(x) + μ^T h(x) + ρ * ‖h(x)‖^2 を定義するである
        sub_prob = lambda x: obj_f(x) + mu_k @ eq_h(x) + rho * pen_val(x)

        # SciPy による無制約最小化である（手法は BFGS を例示）
        result_scipy = optimize.minimize(sub_prob, x_k, method="BFGS")
        x_k = result_scipy.x  # 内側ソルバの解を採用するである

        # ログ出力：目的値 f、制約残差 P、h(x)、現在点 x、乗数 μ
        px = pen_val(x_k)
        hx = eq_h(x_k)
        print(
            "iter = ",
            k + 1,
            ", f(x) = ",
            obj_f(x_k),
            ", P(x) = ",
            px,
            ", \n h(x) = ",
            hx,
            ", x = ",
            x_k,
            ", mu = ",
            mu_k,
        )

        # 収束判定：‖h(x)‖^2 が十分小さければ停止するである
        if px < eps:
            break

        # 乗数更新： μ ← μ + ρ h(x) である
        mu_k = mu_k + rho * hx

        # ペナルティ係数を増加させるである（厳しさを強める）
        rho *= 5.0

    return x_k
