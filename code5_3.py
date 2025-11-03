# ガウス–ニュートン（Gauss–Newton）法の簡潔実装である。
# 目的：非線形最小二乗問題
#          minimize_x  (1/2) * || r(x) ||_2^2
#       を解くための反復更新を与えるである。
# 基本アイデア：目的関数 f(x) = (1/2) * r(x)^T r(x) の
#   勾配     g(x) = J(x)^T r(x)
#   近似ヘッセ B(x) ≈ J(x)^T J(x)    （二次の項 r の二階微分は無視）
# を用い、(J^T J) d = - J^T r を解いて x ← x + d と更新するである。
#
# 収束に関する注意：
# - 残差が十分小さい、あるいはモデルがほぼ線形な領域では良好に収束しやすいである。
# - J^T J が特異/準特異（悪条件）なとき np.linalg.solve は不安定になり得るである。
#   その場合は Levenberg–Marquardt（J^T J + λ I）のダンピングを導入すると堅牢になるである。
# - 終了判定を ||J^T r|| だけでなく、||r|| の減少や ||d|| の小ささでも補強すると実務上は安定である。

import numpy as np
from typing import Callable


def gauss_newton(
    res_r: Callable[
        [np.ndarray], np.ndarray
    ],  # 残差ベクトル r(x) を返す関数である（形状: (m,)）
    jac_r: Callable[
        [np.ndarray], np.ndarray
    ],  # ヤコビ行列 J(x) を返す関数である（形状: (m, n)）
    x_k: np.ndarray,  # 初期点ベクトルである（形状: (n,)）
    eps: float = 1e-6,  # 終了判定に用いるしきい値（||J^T r|| 基準）である
    max_iter: int = 1000,  # 最大反復回数である
) -> np.ndarray:
    """
    Gauss–Newton 法により非線形最小二乗問題を解く関数である。

    反復（k = 0,1,2,...）の流れ：
      1) r_k = r(x_k), J_k = J(x_k)
      2) B_k = J_k^T J_k,  g_k = J_k^T r_k
      3) B_k d_k = - g_k を解く（線形方程式）
      4) x_{k+1} = x_k + d_k
      5) ||g_k|| < eps で停止

    戻り値：近似解 x_k
    """
    for k in range(max_iter):
        # 残差ベクトル r(x_k) とヤコビ行列 J(x_k) を評価するである
        r_k = res_r(x_k)  # 形状 (m,)
        J_k = jac_r(x_k)  # 形状 (m, n)

        # 近似ヘッセと勾配を構成するである
        #   B_k = J_k^T J_k は対称正定（フルランクかつ列独立なら）になりやすい
        #   nab_f_k = g_k = J_k^T r_k は目的 f の勾配である
        B_k = J_k.T @ J_k
        nab_f_k = J_k.T @ r_k

        # 探索方向 d_k を線形方程式から得るである
        # 数値的に不安なら np.linalg.lstsq や 正則化 B_k + λ I の導入を検討するである
        d_k = np.linalg.solve(B_k, -nab_f_k)

        # 点を更新するである（本実装はラインサーチなしの純粋 GN ステップ）
        x_k = x_k + d_k

        # 終了判定：||J^T r|| が十分小さければ停止とするである
        if np.linalg.norm(nab_f_k) < eps:
            break

    # 収束情報の簡易出力（反復回数と残差ノルム）である
    print("GN, iter:", k + 1, "||r(x_k)||:", np.linalg.norm(r_k))
    return x_k
