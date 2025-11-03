# Huschens 法（アルゴリズム 5.3）の実装である。
# 目的：非線形最小二乗問題
#   minimize_x  (1/2) * || r(x) ||_2^2
# に対して，Gauss–Newton の近似ヘッセ J^T J に「低ランク補正 A_k」を加えることで
# 収束性と安定性の改善を図る手法である。
#
# 概要：
#   ・各反復で Gauss–Newton 方向 d_k を B_k d_k = -g_k（B_k = J_k^T J_k + ||r_k|| A_k）から解く。
#   ・補正行列 A_k は Huschens の更新式
#         A_{k+1} = A_k - (B_k s_k)(B_k s_k)^T / (s_k^T B_k s_k)
#                          + (z_k z_k^T) / (s_k^T z_k) * (1 / ||r_{k+1}||)
#     により更新される。
#   ・ここで s_k = x_{k+1} - x_k，z_k = (J_{k+1}^T J_{k+1}) s_k
#       + (J_{k+1} - J_k)^T r_{k+1} * (||r_{k+1}|| / ||r_k||) である。
#
# 数値的注意：
#   ・分母 s_k^T B_k s_k，s_k^T z_k が極小のときに発散し得るため，小さな下限を設ける。
#   ・J^T J が悪条件な場合は np.linalg.solve が不安定になり得る。
#     必要に応じてダンピング（Levenberg–Marquardt: J^T J + λI）を検討するべきである。
#   ・終了判定は ||g_k|| のみでなく，||r_k|| の減少や ||d_k|| の小ささも併用すると堅牢である。

import numpy as np
from typing import Callable


def Huschens(
    res_r: Callable[
        [np.ndarray], np.ndarray
    ],  # 残差ベクトル r(x) を返す関数である（形状: (m,)）
    jac_r: Callable[
        [np.ndarray], np.ndarray
    ],  # ヤコビ行列 J(x) を返す関数である（形状: (m, n)）
    x_k: np.ndarray,  # 初期点ベクトルである（形状: (n,)）
    eps: float = 1e-6,  # 終了判定のしきい値（||g_k|| 基準）である
    max_iter: int = 1000,  # 最大反復回数である
) -> np.ndarray:
    """
    Huschens 法により非線形最小二乗問題を解く関数である。
    返り値は最終点 x_k である。
    """

    n = len(x_k)
    # 低ランク補正のための行列 A_k を 0 初期化するである
    A_k = np.zeros((n, n))

    # 初期ヤコビ行列と残差を評価するである
    J_k = jac_r(x_k)
    r_k = res_r(x_k)

    # Gauss–Newton の基本量を構成するである
    JJ_k = J_k.T @ J_k  # J_k^T J_k
    r_k_norm = np.linalg.norm(r_k)  # ||r_k||
    g_k = J_k.T @ r_k  # 勾配 g_k = J_k^T r_k
    B_k = JJ_k + r_k_norm * A_k  # 近似ヘッセ B_k = J^T J + ||r|| A_k

    # 数値安定用の極小下限である
    denom_eps = 1e-12

    for k in range(max_iter):
        # 探索方向を解くである： B_k d_k = - g_k
        # 必要に応じて正則化（B_k += λI）を追加するとよい。
        d_k = np.linalg.solve(B_k, -g_k)

        # 1 ステップ更新し，新しい点・ヤコビ行列・残差を評価するである
        x_old, J_old, r_old_norm = x_k, J_k, r_k_norm
        x_k = x_k + d_k
        J_k = jac_r(x_k)
        r_k = res_r(x_k)

        # 新しい基本量を計算するである
        JJ_k = J_k.T @ J_k
        r_k_norm = np.linalg.norm(r_k)
        g_k = J_k.T @ r_k

        # s_k と z_k を構成するである
        s_k = x_k - x_old
        # z_k = (J_{k+1}^T J_{k+1}) s_k + (J_{k+1} - J_k)^T r_{k+1} * (||r_{k+1}|| / ||r_k||)
        #      第2項でヤコビ行列差分が現れる点が Gauss–Newton と異なる補正である。
        J_diff_T_r = (J_k - J_old).T @ r_k
        scale = r_k_norm / max(r_old_norm, denom_eps)
        z_k = JJ_k @ s_k + J_diff_T_r * scale

        # B_k を再構成するである（A_k は現時点でのものを使用）
        B_k = JJ_k + r_k_norm * A_k

        # A_k を更新するである： A_{k+1} = A_k - (Bs Bs^T)/(s^T Bs) + (z z^T)/(s^T z) * (1/||r||)
        Bs = B_k @ s_k
        sTBs = float(s_k @ Bs)
        sTz = float(s_k @ z_k)

        # 分母に極小下限を設けて発散を防止するである
        if abs(sTBs) < denom_eps or abs(sTz) < denom_eps or r_k_norm < denom_eps:
            # 数値不安定が強い場合は補正をスキップし，次反復で安定化を図るである
            # （必要に応じてここで B_k に小さな対角正則化を入れる選択肢もある）
            pass
        else:
            A_k = A_k - np.outer(Bs, Bs) / sTBs + np.outer(z_k, z_k) / sTz / r_k_norm

        # 次反復に向けて B_k を更新しておくである
        B_k = JJ_k + r_k_norm * A_k

        # 終了判定：||g_k|| が十分小さければ停止するである
        if np.linalg.norm(g_k) < eps:
            break

    # 結果の簡易出力である
    print("Huschens, iter:", k + 1, "||r(x_k)||:", np.linalg.norm(r_k))
    return x_k
