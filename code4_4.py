"""
コード4.4 ■ BFGS 公式（B 公式）のコード（説明コメント付き）
本関数はヘッセ近似 B_k ≈ ∇²f(x_k) を BFGS（B 公式）で更新するである。
更新式は次の通りである：
    B_{k+1} = B_k - \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k} + \frac{y_k y_k^T}{s_k^T y_k},
ここで s_k = x_{k+1}-x_k,  y_k = ∇f(x_{k+1}) - ∇f(x_k) である。
性質：B_k が対称正定かつ曲率条件 s_k^T y_k > 0 を満たせば、B_{k+1} も正定を保つである。
実装上の要点：
- 分母 s^T y（曲率条件）が小さい/非正だと不安定になるためガードが必要である。
- 浮動小数点誤差で非対称が混入し得るため、更新後の対称化が有効である。
"""

import numpy as np


def BFGS_B(
    B: np.ndarray,
    s: np.ndarray,
    y: np.ndarray,
    *,
    eps: float = 1e-12,
    symmetrize: bool = True
) -> np.ndarray:
    """
    BFGS（B 公式）によりヘッセ近似 B を更新するである。

    Parameters
    ----------
    B : (n, n) ndarray
        現在のヘッセ近似 B_k（対称正定を想定）である。
    s : (n,) ndarray
        変化量 s_k = x_{k+1} - x_k である。
    y : (n,) ndarray
        勾配差分 y_k = ∇f(x_{k+1}) - ∇f(x_k) である。
    eps : float, optional
        曲率量 s^T y の下限ガードである（デフォルト 1e-12）。
    symmetrize : bool, optional
        更新後に (B + B^T)/2 で対称化するかどうかである。

    Returns
    -------
    (n, n) ndarray
        更新後のヘッセ近似 B_{k+1} である。
    """

    # --- 形状の簡易チェックである（開発時のバグ検知に有効） ---
    n = B.shape[0]
    assert B.shape == (n, n), "B は正方行列である必要があるである"
    assert s.shape == (n,), "s の形状は (n,) である必要があるである"
    assert y.shape == (n,), "y の形状は (n,) である必要があるである"

    # --- 曲率条件（s^T y > 0）の確認とガードである ---
    sy = float(s @ y)
    if sy <= eps:
        # 曲率条件が不十分な場合は更新をスキップする保守的設計である
        # （代替として Powell のダンピング等を導入することも多い）
        return B if not symmetrize else 0.5 * (B + B.T)

    # --- BFGS(B) 更新の各項を計算するである ---
    Bs = B @ s
    sBs = float(s @ Bs)  # 分母 s^T B s は > 0 が望ましい（B 正定を仮定）

    # 数値安定化：s^T B s が極小なら更新を控えるである
    if sBs <= eps:
        return B if not symmetrize else 0.5 * (B + B.T)

    # 本体更新：
    #  第1補正項：-(B s s^T B)/(s^T B s) は「古い曲率」を除去する役割である。
    #  第2補正項：+(y y^T)/(s^T y) は「新しい曲率」を反映する役割である。
    B_new = B - np.outer(Bs, Bs) / sBs + np.outer(y, y) / sy

    # --- 対称化（丸め誤差対策）である ---
    if symmetrize:
        B_new = 0.5 * (B_new + B_new.T)

    return B_new
