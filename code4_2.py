"""
コード4.2 ■ BFGS 公式（H 公式）のコード（説明コメント付き）
本関数は、**逆ヘッセ近似** H_k ≈ (∇^2 f(x_k))^{-1} を BFGS 公式で更新するものである。

更新式（BFGS の H 公式）は次である：
    H_{k+1} = H_k
              - (H_k y_k s_k^T + s_k y_k^T H_k) / (s_k^T y_k)
              + (1 + (y_k^T H_k y_k) / (s_k^T y_k)) * (s_k s_k^T) / (s_k^T y_k)

ここで、s_k = x_{k+1} - x_k、y_k = ∇f(x_{k+1}) - ∇f(x_k) である。
BFGS は 1) 対称性、2) 正定性（条件 s_k^T y_k > 0 の下）を保つ更新として知られている。

実装上の注意：
- 分母 s^T y は**曲率条件**であり、これが正（十分に大きい）でないと数値不安定や正定性喪失が起き得る。
- 実務では Powell のダンピングや閾値クリップで s^T y を下駄履きすることが多い。
- 浮動小数点誤差により非対称が生じ得るため、更新後に (H + H^T)/2 で対称化すると堅牢である。

Parameters
----------
H : np.ndarray (n, n)
    現在の逆ヘッセ近似 H_k（対称正定値を想定）。
s : np.ndarray (n,)
    変化量ベクトル s_k = x_{k+1} - x_k。
y : np.ndarray (n,)
    勾配差分ベクトル y_k = ∇f(x_{k+1}) - ∇f(x_k)。

Returns
-------
H_new : np.ndarray (n, n)
    BFGS 更新後の逆ヘッセ近似 H_{k+1} である。
"""

import numpy as np


def BFGS_H(
    H: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    *,
    symmetrize: bool = True,
    eps: float = 1e-12
) -> np.ndarray:
    # --- 形状チェック（開発時のバグ検知用）である -------------------------
    n = H.shape[0]
    assert H.shape == (n, n), "H は正方行列である必要があるである"
    assert s.shape == (n,), "s の形状は (n,) である必要があるである"
    assert y.shape == (n,), "y の形状は (n,) である必要があるである"

    # --- 曲率量 s^T y を計算するである -----------------------------------
    sy = float(s @ y)

    # 数値安定化：s^T y が極端に小さい/非正のときのガード
    # 典型的には sy > 0 が望ましい。非正の場合、更新をスキップするかダンピングを行う。
    if sy <= eps:
        # 最小限の対処：更新を行わず現在の H を返す（安全志向）
        # 実運用では Powell damping 等の導入を検討するである。
        return H if not symmetrize else 0.5 * (H + H.T)

    # --- 中間量を計算するである ------------------------------------------
    Hy = H @ y  # H_k y_k
    yHy = float(y @ Hy)  # y_k^T H_k y_k

    # --- BFGS（H 公式）で更新するである -----------------------------------
    # 第一項： H そのもの
    H_new = H.copy()

    # 第二項：-(H y s^T + s y^T H) / (s^T y)
    H_new -= (np.outer(Hy, s) + np.outer(s, Hy)) / sy

    # 第三項： (1 + (y^T H y) / (s^T y)) * (s s^T) / (s^T y)
    coef = (1.0 + yHy / sy) / sy
    H_new += coef * np.outer(s, s)

    # --- 対称化（丸め誤差対策）である -------------------------------------
    if symmetrize:
        H_new = 0.5 * (H_new + H_new.T)

    return H_new
