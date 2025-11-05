"""
コード7.4 ■ SQP 法（アルゴリズム7.4）＋ Powell 減衰 BFGS(B) 更新の実装（説明コメント付き）である。

目的：
  一般の非線形制約付き最適化問題
      minimize   f(x)
      s.t.       g(x) ≤ 0   （不等式制約，m本）
                 h(x) = 0   （等式制約，ℓ本）
  を逐次二次計画法（SQP）で解く。各反復で二次計画（QP）サブ問題を解き、
  方向 Δx を得て、メリット関数に対する Armijo 型直線探索でステップ幅を決め、
  準ニュートン近似 B を Powell の減衰 BFGS(B) 公式で更新する。

数式要点（概略）：
  ・QP サブ問題（Newton 近似の 2 次モデル＋制約の 1 次近似）
      minimize   (1/2) Δx^T B_k Δx + ∇f(x_k)^T Δx
      s.t.       g(x_k) + ∇g(x_k)^T Δx ≤ 0
                 h(x_k) + ∇h(x_k)^T Δx = 0
  ・Lagrange 勾配の差分から y_k を構成（Powell 減衰 BFGS 用）
      y_k = [∇f(x_{k+1}) − ∇f(x_k)]
            + [∇g(x_{k+1}) − ∇g(x_k)] λ_k
            + [∇h(x_{k+1}) − ∇h(x_k)] μ_k
  ・メリット関数（l1 型正確ペナルティ）
      Φ_ρ(x) = f(x) + ρ (‖max(0, g(x))‖_1 + ‖h(x)‖_1)
    その 1 次近似 Φ_ρ^lin を用いた Armijo 条件でステップ幅 α を選ぶ。

実装メモ：
  - QP は cvxpy で解く（dual_value から λ, μ を取得）。
  - ρ は λ, μ の大きさから動的に決める（典型的に ρ ≥ 2 max(|λ_i|, |μ_j|)）。
  - Powell 減衰により B の正定性を保ちやすくする（BFGS_Powell を使用）。
  - 数値ガード（極小ステップで打ち切り、B の対称化など）を適宜入れる。
"""

import numpy as np
import cvxpy as cp
from typing import Callable


# --- Powell 減衰 BFGS(B) 更新（B 行列の更新）を用意する（依存関数） ---
def BFGS_Powell(
    B: np.ndarray, s: np.ndarray, y: np.ndarray, omega: float = 0.2
) -> np.ndarray:
    """
    Powell の減衰 BFGS(B) 更新である：
      B_{+} = B - (B s s^T B)/(s^T B s) + (z z^T)/(s^T z),
      z = ψ y + (1-ψ) (B s),  ψ∈[0,1]
    ここでは sy と s^T B s の比較から ψ を決める。
    """
    sy = float(s @ y)
    Bs = B @ s
    sBs = float(s @ Bs)
    eps = 1e-12
    if sBs <= eps:
        return B
    denom = sBs - sy
    if sy >= omega * sBs or abs(denom) < eps:
        psi = 1.0
    else:
        psi = (1.0 - omega) * sBs / denom
        psi = min(1.0, max(0.0, psi))
    z = psi * y + (1.0 - psi) * Bs
    sTz = float(s @ z)
    if abs(sTz) < eps:
        return B
    B_new = B - np.outer(Bs, Bs) / sBs + np.outer(z, z) / sTz
    # 対称化（丸め誤差対策）
    return 0.5 * (B_new + B_new.T)


def SQP(
    obj_f: Callable[[np.ndarray], float],  # f(x)
    ineq_g: Callable[[np.ndarray], np.ndarray],  # g(x) ∈ R^m
    eq_h: Callable[[np.ndarray], np.ndarray],  # h(x) ∈ R^ℓ
    nab_f: Callable[[np.ndarray], np.ndarray],  # ∇f(x) ∈ R^n
    nab_g: Callable[
        [np.ndarray], np.ndarray
    ],  # ∇g(x) の Jacobian（形状 (m, n) を想定）を返す関数
    nab_h: Callable[
        [np.ndarray], np.ndarray
    ],  # ∇h(x) の Jacobian（形状 (ℓ, n) を想定）を返す関数
    x_k: np.ndarray,  # 初期点 ∈ R^n
    max_iter: int = 100,
    xi: float = 0.1,  # Armijo 条件の係数（0 < xi < 1）
    eps: float = 1e-8,  # 終了判定閾値（‖Δx‖）
) -> np.ndarray:
    """
    SQP（逐次二次計画法）で制約付き最適化を解くである。
    返り値は最終反復の近似解 x_k である。
    """

    # --- 問題サイズと初期化 ---
    n = len(x_k)
    m = len(ineq_g(x_k))
    l = len(eq_h(x_k))
    B_k = np.eye(n)  # ラグランジアンのヘッセ近似（準ニュートンで更新）
    lamb = np.zeros(m)  # 不等式制約の双対変数（λ）
    mu = np.zeros(l)  # 等式制約の双対変数（μ）

    # --- メリット関数 Φ_ρ とその 1 次近似 Φ_ρ^lin を定義 ---
    def merit_func(x: np.ndarray, rho: float) -> float:
        g_vals = ineq_g(x).astype(float, copy=False)
        h_vals = eq_h(x).astype(float, copy=False)
        g_pos = np.maximum(0.0, g_vals)  # max(0, g)
        return float(obj_f(x) + rho * (np.sum(g_pos) + np.sum(np.abs(h_vals))))

    def merit_func_l(x: np.ndarray, dx: np.ndarray, rho: float) -> float:
        # g(x) + ∇g(x_k) dx と h(x) + ∇h(x_k) dx の 1 次近似
        gx_lin = ineq_g(x) + nab_g(x_k).T @ dx
        hx_lin = eq_h(x) + nab_h(x_k).T @ dx
        gx_lin = np.maximum(0.0, gx_lin)
        return float(obj_f(x) + rho * (np.sum(gx_lin) + np.sum(np.abs(hx_lin))))

    # --- 反復 ---
    for k in range(max_iter):
        # 1) QP サブ問題の定式化（cvxpy で解く）
        delta_x = cp.Variable(n)
        # 目的：(1/2) Δx^T B_k Δx + ∇f(x_k)^T Δx
        objective = 0.5 * cp.quad_form(delta_x, B_k) + nab_f(x_k).T @ delta_x
        # 制約：g(x_k) + ∇g(x_k)^T Δx ≤ 0,  h(x_k) + ∇h(x_k)^T Δx = 0
        A_g = nab_g(
            x_k
        ).T  # 形状 (n, m) の転置 → (m, n) に注意：nab_g は (m, n) を返す前提
        A_h = nab_h(x_k).T
        constraints = [ineq_g(x_k) + A_g @ delta_x <= 0, eq_h(x_k) + A_h @ delta_x == 0]
        subQP = cp.Problem(cp.Minimize(objective), constraints)

        # 2) QP を解く（ソルバは環境依存。既定でも良いが OSQP/ECOS/SCS 等を指定可）
        subQP.solve()  # 失敗時は subQP.status を確認すること
        delta_x_k = np.array(delta_x.value).reshape(-1)

        # dual_value から λ, μ を更新（cvxpy の仕様で符号/形が環境依存の場合あり）
        # ここでは典型的に不等式→≤ 制約の双対は ≥0、等式の双対は符号自由を想定
        lamb = np.array(subQP.constraints[0].dual_value).reshape(-1)
        mu = np.array(subQP.constraints[1].dual_value).reshape(-1)

        # 3) 終了判定（方向が十分小さい）
        if np.linalg.norm(delta_x_k) < eps:
            break

        # 4) メリット関数の係数 ρ を設定（λ, μ の絶対値の最大に比例：ρ ≥ 2 max(|λ|, |μ|)）
        #   np.max を配列に直接かけると配列返しになりうるので、スカラーに正規化する。
        max_dual = max(
            float(np.max(np.abs(lamb))) if lamb.size else 0.0,
            float(np.max(np.abs(mu))) if mu.size else 0.0,
        )
        rho = max(1.0, 2.0 * max_dual)

        # 5) Armijo 型直線探索（l1 正確ペナルティの 1 次近似を用いる）
        x_old = x_k.copy()
        P_old = merit_func(x_old, rho)
        delta_P = merit_func_l(x_old, delta_x_k, rho) - P_old  # 1 次減少量
        alpha = 1.0
        # 方向が非減少の場合のガード（delta_P ≥ 0 のときは α をすぐ縮小）
        if delta_P >= 0:
            alpha = 0.5

        # Armijo 条件：Φ(x+αΔx) ≤ Φ(x) + ξ α ΔΦ_lin
        while merit_func(x_old + alpha * delta_x_k, rho) > P_old + xi * alpha * delta_P:
            alpha *= 0.5
            if alpha < 1e-5:
                # ステップ幅が極小化したら打ち切る（実務では方向再計算等の分岐を入れる）
                break

        # 6) x を更新
        x_k = x_old + alpha * delta_x_k

        # 7) B の更新（Powell 減衰 BFGS(B)）
        s_k = x_k - x_old
        # y_k = ∇L(x_{k+1},λ,μ) − ∇L(x_k,λ,μ) の近似
        #     = [∇f(x_{k+1}) − ∇f(x_k)]
        #       + [∇g(x_{k+1}) − ∇g(x_k)] λ
        #       + [∇h(x_{k+1}) − ∇h(x_k)] μ
        y_k = (
            (nab_f(x_k) - nab_f(x_old))
            + (nab_g(x_k) - nab_g(x_old)).T @ lamb
            + (nab_h(x_k) - nab_h(x_old)).T @ mu
        )

        B_k = BFGS_Powell(B_k, s_k, y_k)

        # 進捗出力
        print(
            "iter = ",
            k + 1,
            ", f(x) = ",
            obj_f(x_k),
            ", g(x) = ",
            ineq_g(x_k),
            ", h(x) = ",
            eq_h(x_k),
            ", \n x = ",
            x_k,
            ", lambda = ",
            lamb,
            ", mu = ",
            mu,
        )

    return x_k
