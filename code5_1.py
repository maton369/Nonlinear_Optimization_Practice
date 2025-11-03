# L-BFGS における two-loop recursion（アルゴリズム5.1）の実装である。
# 目的：過去 t 個の曲率ペア (s_i, y_i) と初期逆ヘッセ近似 H0 を用いて，
#       現在の勾配ベクトル q に対する近似ニュートン方向 r = H_k * q を計算するである。
#
# 背景（L-BFGS の要点）：
#   - BFGS(H) の更新を明示的に畳み重ねる代わりに，過去の差分 s_i = x_{i+1}-x_i，
#     y_i = ∇f(x_{i+1})-∇f(x_i) と ρ_i = 1/(y_i^T s_i) を保存しておき，
#     2 本のループ（後退→前進）で H_k q を O(mt) + O(n) で求める（m は保持メモリ，n は次元）である。
#   - Step 1（後退ループ）では q から y 成分を順に落としていき，係数 α_i を蓄える。
#   - Step 2 で初期近似 H0（通常は γ_k I）を掛ける。γ_k はスケーリング係数である。
#   - Step 3（前進ループ）で s 成分を足し戻し，方向 r を完成させる。
#
# 変数の約束：
#   q   : (n,) 勾配ベクトル g_k など，H_k を掛けたいベクトル（通常は ∇f(x_k)）である。
#   S   : (n, t) 列に s_0..s_{t-1} を積んだ行列である。
#   Y   : (n, t) 列に y_0..y_{t-1} を積んだ行列である。
#   rho : (t,)  各列ペアに対応する ρ_i = 1 / (y_i^T s_i) である（曲率条件により正を想定）。
#   H0  : float または (n,n) ndarray。初期逆ヘッセ近似で，γ_k I を想定するならスカラーで良い。
#   t   : 使用するペア数（0 ≤ t ≤ S.shape[1]）。新しい順・古い順の取り扱いは呼び出し側の責務である。
#
# 数値上の注意：
#   - ρ_i の計算時に y_i^T s_i が極小/非正だと不安定になる。L-BFGS 登録段階でガードするのが定石である。
#   - H0 は一般に γ_k = (s_{t-1}^T y_{t-1}) / (y_{t-1}^T y_{t-1}) を用いるスケーリングが推奨である。
#   - t=0（メモリなし）の場合，r = H0 * q（= γ_k q）となる。
#
# 返り値：
#   r : (n,) 近似ニュートン方向 H_k q（通常は -H_k ∇f で降下方向を取るため，呼び出し側で符号を調整する）である。

import numpy as np
from typing import Union


def two_loop(
    q: np.ndarray,
    S: np.ndarray,
    Y: np.ndarray,
    rho: np.ndarray,
    H0: Union[float, np.ndarray],
    t: int,
) -> np.ndarray:
    """
    L-BFGS two-loop recursion により r = H_k q を計算するである。

    Parameters
    ----------
    q : (n,) ndarray
        入力ベクトル（通常は現ステップの勾配 g_k）である。
    S : (n, t_max) ndarray
        過去の s 列ベクトルを列結合した行列である（右端が最も新しいとは限らない）。
    Y : (n, t_max) ndarray
        過去の y 列ベクトルを列結合した行列である。
    rho : (t_max,) ndarray
        各ペアの ρ_i = 1/(y_i^T s_i)。登録時に数値ガード済みを想定する。
    H0 : float or (n,n) ndarray
        初期逆ヘッセ近似。スカラーの場合は γ_k I を表す。
    t : int
        使用する最新 t ペア数。S, Y, rho の先頭から t 本を用いる想定である。

    Returns
    -------
    r : (n,) ndarray
        出力ベクトル r = H_k q である。
    """
    n = q.shape[0]
    # --- 形状の簡易チェックである（開発時のバグ検知用） ---
    assert S.shape[0] == n and Y.shape[0] == n, "S, Y の行数は q と一致させるである"
    assert (
        S.shape[1] >= t and Y.shape[1] >= t and rho.shape[0] >= t
    ), "t が保持本数を超えているである"
    # 内部で q を破壊的更新するためコピーを取るである
    q = q.astype(float, copy=True)

    # Step 1: 後退ループ（i = 0..t-1 の順に α_i を求め q を更新する）
    #  α_i = ρ_i s_i^T q,  q ← q - α_i y_i
    a = np.zeros(t, dtype=float)
    for i in range(t):
        si = S[:, i]
        yi = Y[:, i]
        ai = rho[i] * np.dot(si, q)  # α_i = ρ_i s_i^T q
        a[i] = ai
        q -= ai * yi  # q ← q - α_i y_i

    # Step 2: 初期近似 H0 を適用
    #  r = H0 q （H0 がスカラーなら γ_k q，行列なら行列積）
    if np.isscalar(H0):
        r = float(H0) * q
    else:
        # H0 が (n,n) の場合は行列積である
        r = H0 @ q

    # Step 3: 前進ループ（i = t-1..0 の順に β を用いて r を更新）
    #  β_i = ρ_i y_i^T r,  r ← r + s_i (α_i - β_i)
    for i in range(t - 1, -1, -1):
        yi = Y[:, i]
        si = S[:, i]
        bi = rho[i] * np.dot(yi, r)  # β_i = ρ_i y_i^T r
        r += si * (a[i] - bi)  # r ← r + s_i(α_i - β_i)

    # Step 4: r を返す（= H_k q）
    return r
