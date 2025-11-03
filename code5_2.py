"""
コード5.2 ■ 記憶制限付き BFGS 法（L-BFGS：アルゴリズム5.2） の実装（説明コメント付き）
目的：
  L-BFGS により、過去 m (= memory) 本の曲率情報 (s_i, y_i) のみを保持しつつ、
  two-loop recursion で近似逆ヘッセ作用 H_k * g を計算して探索方向 d_k を得るである。
  ストレージ O(n m)、一回の更新コスト O(n m) で、準ニュートン級の収束性を実現する。

設計の要点：
  - ラインサーチは Wolfe 条件を満たすもの（line_Wolfe）を想定するである。
  - two_loop は r = H_k q を返す（通常は q = -∇f で降下方向 r を得る）である。
  - 曲率条件 s^T y > 0 を満たす場合のみ (s, y) を履歴に登録して数値安定性を確保するである。
  - 初期逆ヘッセ近似 H0 は γ_k I（スカラー γ_k）のスケーリングを採用するである。
    γ_k = (s_{last}^T y_{last}) / (y_{last}^T y_{last}) が定石である。

返り値：
  - 最終の点 x_k を返す（点列が必要ならロギング版を別関数で用意するのが良いである）。
"""

import numpy as np
from typing import Tuple
from code5_1 import two_loop  # two-loop recursion を使用するである
from code2_2 import line_Wolfe  # Wolfe 条件ベースの線探索を使用するである


def _push_memory(
    S: np.ndarray, Y: np.ndarray, rho: np.ndarray, s: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    L-BFGS の記憶（S, Y, rho）を 1 本分ローテーションして先頭に (s, y, 1/(y^T s)) を追加するである。
    曲率条件 s^T y > 0 を満たさない場合は追加をスキップするである。

    Parameters
    ----------
    S, Y : (n, m) ndarray
        それぞれ s 列と y 列を列方向に積んだ行列である（右に行くほど古いものを入れる実装にする）。
    rho : (m,) ndarray
        ρ_i = 1 / (y_i^T s_i) の履歴である。
    s, y : (n,) ndarray
        新たに得た曲率ペアである。

    Returns
    -------
    S, Y, rho : 更新後の記憶配列である。
    accepted : bool
        追加を受理したか（曲率条件を満たしたか）である。
    """
    sy = float(s @ y)
    if sy <= 1e-12:  # 曲率条件ガード（極小・非正は不安定のため採用しない）
        return S, Y, rho, False

    # 1 本右へロールし、先頭に新規ペアを詰めるである
    S = np.roll(S, 1, axis=1)
    Y = np.roll(Y, 1, axis=1)
    rho = np.roll(rho, 1)

    S[:, 0] = s
    Y[:, 0] = y
    rho[0] = 1.0 / sy
    return S, Y, rho, True


def _init_H0_scalar(S: np.ndarray, Y: np.ndarray) -> float:
    """
    two-loop で用いる初期逆ヘッセ近似 H0 をスカラー γI として返すである。
    原則として「最新」のペア (s_0, y_0) から
      γ = (s_0^T y_0) / (y_0^T y_0)
    を採用する。初期化直後などで使用可能なペアがない場合は γ=1.0 とするである。
    """
    s0 = S[:, 0]
    y0 = Y[:, 0]
    num = float(s0 @ y0)
    den = float(y0 @ y0)
    if den <= 1e-12 or num <= 0.0:  # 数値ガード（曲率条件 or 0 除算）
        return 1.0
    return num / den


def LBFGS_H(
    obj_f,  # callable: f(x) を返す
    nab_f,  # callable: ∇f(x) を返す
    x_k: np.ndarray,  # 初期点 (n,)
    *,
    max_iter: int = 1000,
    eps: float = 1.0e-8,
    memory: int = 10
) -> np.ndarray:
    """
    記憶制限付き BFGS（L-BFGS：H 公式による two-loop 利用）で最小化を行うである。

    Parameters
    ----------
    obj_f : callable
        目的関数 f(x) を返す関数である。
    nab_f : callable
        勾配 ∇f(x) を返す関数である。
    x_k : np.ndarray (n,)
        初期点である。
    max_iter : int
        最大反復回数である。
    eps : float
        勾配ノルムによる終了判定閾値である。
    memory : int
        L-BFGS が保持する過去ペア本数 m である（典型的には 5〜20）。

    Returns
    -------
    np.ndarray
        最終点 x_k である。
    """

    n = len(x_k)
    # --- 記憶領域の初期化（S, Y は「列にペアを積む」形。rho は各列に対応） ---
    S_k = np.zeros((n, memory))  # s の履歴（列ベクトル）
    Y_k = np.zeros((n, memory))  # y の履歴（列ベクトル）
    rho_k = np.zeros(memory)  # ρ の履歴
    m_k = 0  # 現在の有効本数（<= memory）

    # --- 初期勾配と初期方向（最急降下方向） ---
    nab_f_k = nab_f(x_k)
    d_k = -nab_f_k

    for k in range(max_iter):
        # 収束判定：‖∇f(x_k)‖ ≤ eps
        if np.linalg.norm(nab_f_k) <= eps:
            break

        # 1) Wolfe 線探索でステップ幅 α を得るである
        alpha = line_Wolfe(obj_f, nab_f, x_k, d_k)

        # 2) 点の更新
        x_old = x_k
        x_k = x_k + alpha * d_k

        # 3) 勾配の更新と曲率ペアの生成
        g_old = nab_f_k
        nab_f_k = nab_f(x_k)
        s_k = x_k - x_old
        y_k = nab_f_k - g_old

        # 4) 曲率ペアの記憶に登録（曲率条件 s^T y > 0 を満たす場合のみ）
        S_k, Y_k, rho_k, accepted = _push_memory(S_k, Y_k, rho_k, s_k, y_k)
        if accepted:
            m_k = min(m_k + 1, memory)  # 有効本数を増やす

        # 5) 初期逆ヘッセ近似 H0 = γ I（利用可能な場合）
        H0 = _init_H0_scalar(S_k, Y_k) if m_k > 0 else 1.0

        # 6) two-loop で近似ニュートン方向を計算：d_k = - H_k ∇f(x_k)
        #    最新から m_k 本を使用（S_k, Y_k は先頭が最新という前提で two_loop を呼ぶ）
        d_k = two_loop(-nab_f_k, S_k, Y_k, rho_k, H0, m_k)

    print("L-BFGS, iter:", k + 1, "f(x):", obj_f(x_k))
    return x_k
