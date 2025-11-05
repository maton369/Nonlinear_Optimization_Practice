# 外点ペナルティ関数法（Exterior Penalty Method）の実装である。
# 目的：
#   制約付き最適化問題
#       minimize   f(x)
#       subject to g_i(x) ≤ 0  (i = 1,...,m)   （不等式制約）
#                  h_j(x) = 0  (j = 1,...,ℓ)   （等式制約）
#   を、逐次的に「無制約最適化問題」
#       minimize   f(x) + ρ_k * P(x)
#   に置き換えて解くである。ここで P(x) は制約違反の大きさを測るペナルティである。
#
# アルゴリズムの骨子（アルゴリズム7.2）：
#   1) 初期点 x_0 と初期ペナルティ係数 ρ > 0 を与える。
#   2) 無制約問題  φ_k(x) = f(x) + ρ * P(x) を（内側ソルバで）最小化して x_{k+1} を得る。
#   3) もし P(x_{k+1}) が十分小さければ停止。小さくなければ ρ ← cρ （c > 1）として 2) へ戻る。
#
# この実装では
#   P(x) = ||max(0, g(x))||_2^2 + ||h(x)||_2^2
# を用いる。max(0, g) によって「不等式制約が満たされている部分（g≤0）」はペナルティ0、
# 破っている部分（g>0）のみ二乗で罰するである。
#
# 実装メモ：
#   - ineq_g(x), eq_h(x) はそれぞれ R^n → R^m, R^n → R^ℓ のベクトル値関数を想定するである。
#   - SciPy の minimize を「内側ソルバ」として用い、各外側反復で無制約最適化を行うである。
#   - 元コードでは g_values を in-place で max 処理していたが、副作用を避けるため
#     np.maximum を用いて安全にベクトル化するである。
#   - ρ の倍率（ここでは 5）や外側反復回数は問題依存で調整するである。
#
# 参考：外点法は実装容易だが、ρ が大きくなると内側問題が悪条件化しやすい。
#       実務ではバリア法や増大ラグランジュ法（ALM）も検討に値するである。

import numpy as np
from scipy import optimize
from typing import Callable


def out_penalty_method(
    obj_f: Callable[[np.ndarray], float],  # 目的関数 f(x) を返す
    ineq_g: Callable[
        [np.ndarray], np.ndarray
    ],  # 不等式制約ベクトル g(x) を返す（形状: (m,)）
    eq_h: Callable[
        [np.ndarray], np.ndarray
    ],  # 等式制約ベクトル h(x) を返す（形状: (ℓ,)）
    x_k: np.ndarray,  # 初期点
    rho: float = 1.0,  # 初期ペナルティ係数 ρ
    max_iter: int = 10,  # 外側反復回数の上限
    eps: float = 1e-8,  # ペナルティ値 P(x) に対する停止閾値
) -> np.ndarray:
    """
    外点ペナルティ関数法により制約付き最適化を解くである。
    返り値は最終反復の近似解 x_k である。
    """

    # 不等式・等式の本数をあらかじめ評価しておくである
    m = len(ineq_g(x_k))
    l = len(eq_h(x_k))

    # --- ペナルティ関数 P(x) の定義（ベクトル化・副作用なし） ---
    def pen_val(x: np.ndarray) -> float:
        g_vals = ineq_g(x).astype(float, copy=False)  # g(x)
        h_vals = eq_h(x).astype(float, copy=False)  # h(x)
        g_pos = np.maximum(0.0, g_vals)  # max(0, g_i(x)) を要素毎に
        # 2-ノルム二乗の和を返す（= ベクトル二乗和）
        return float(np.sum(g_pos**2) + np.sum(h_vals**2))

    # 外側反復（ρ を増大させつつ内側問題を解く）
    for k in range(max_iter):
        # 無制約の部分問題 φ(x) = f(x) + ρ * P(x)
        sub_prob = lambda x: obj_f(x) + rho * pen_val(x)

        # SciPy で内側最適化を行うである
        # method はデフォルトに委ねてもよいが、BFGS を明示することもある。
        result_scipy = optimize.minimize(sub_prob, x_k, method="BFGS")
        x_k = result_scipy.x  # 内側ソルバの解を採用

        # ログ出力（解析用）：目的・ペナルティ・制約残差・現在点
        g_now = ineq_g(x_k)
        h_now = eq_h(x_k)
        print(
            f"iter = {k+1}, f(x) = {obj_f(x_k):.6e}, P(x) = {pen_val(x_k):.6e},\n"
            f"  g(x) = {g_now}, h(x) = {h_now},\n"
            f"  x = {x_k}"
        )

        # ペナルティが十分小さければ停止
        if pen_val(x_k) < eps:
            break

        # ペナルティ係数を増大させる（厳しさを上げる）
        rho *= 5.0

    return x_k
