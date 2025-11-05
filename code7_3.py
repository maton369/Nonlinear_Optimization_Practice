import numpy as np


def BFGS_Powell(
    B: np.ndarray, s: np.ndarray, y: np.ndarray, omega: float = 0.2
) -> np.ndarray:
    """
    Powell の修正（damped）BFGS 更新である。
    目的：曲率条件 s^T y > 0 が弱い／破れる場合でも、近似ヘッセ B の正定性を保つ（または損ねにくくする）ための
          減衰補正式を用いた BFGS(B) 更新を行うである。

    標準 BFGS(B) 公式：
        B_{k+1} = B_k - (B_k s s^T B_k) / (s^T B_k s) + (y y^T) / (s^T y)

    Powell の減衰の考え方（代表的な形）：
        まず s^T y と s^T B s を比較し、psi ∈ [0,1] を決める。
        ついで z = psi * y + (1 - psi) * (B s) として、
        y の代わりに z を用いた準 BFGS 更新
            B_{k+1} = B - (B s s^T B)/(s^T B s) + (z z^T)/(s^T z)
        を行うである。
        これにより s^T z が十分に正であるように調整し、更新の安定性・正定性維持を図るである。

    パラメータ：
        B : (n, n) 対称（準）正定なヘッセ近似行列である。
        s : (n,)  直前ステップの変化量 s = x_{k+1} - x_k である。
        y : (n,)  勾配差分 y = ∇f_{k+1} - ∇f_k である。
        omega : 0 < omega < 1 の調整パラメータで、s^T y と s^T B s の比較に用いるである。
                典型値は 0.1〜0.2 程度が用いられることが多いである。

    数値安定化：
        - s^T B s, s^T z が極小の場合は更新をスキップ（元の B を返す）する簡易ガードを入れるである。
        - 更新後は（丸め誤差対策として）対称化 B ← (B + B^T)/2 を行うである。

    戻り値：
        更新後の行列 B_{new} を返すである。
    """
    # --- 基本量の計算である ---
    sy = float(s @ y)  # s^T y
    Bs = B @ s  # B s
    sBs = float(s @ Bs)  # s^T B s

    # --- 数値ガード（分母ゼロや極小を避ける）である ---
    eps = 1e-12
    if sBs <= eps:
        # s^T B s が極小/非正なら安定更新が困難であるため、ここでは更新をスキップするである
        return B

    # --- Powell の減衰係数 psi を決定するである ---
    # 典型的な規則： sy >= omega * sBs のときは psi=1（減衰なし）、
    # そうでなければ psi = (1-omega)*sBs / (sBs - sy) として y を Bs に寄せるである。
    denom = sBs - sy
    if sy >= omega * sBs or abs(denom) < eps:
        psi = 1.0
    else:
        psi = (1.0 - omega) * sBs / denom
        # psi は [0,1] にクリップしておくと過度の減衰を防げるである
        if psi < 0.0:
            psi = 0.0
        elif psi > 1.0:
            psi = 1.0

    # --- 減衰後の疑似 y ベクトル z を構成するである ---
    z = psi * y + (1.0 - psi) * Bs
    sTz = float(s @ z)
    if abs(sTz) < eps:
        # 分母が極小なら更新せずに返すである
        return B

    # --- Powell 修正 BFGS(B) 更新である ---
    # B_new = B - (B s s^T B)/(s^T B s) + (z z^T)/(s^T z)
    Bs_outer = np.outer(Bs, Bs) / sBs
    zz_outer = np.outer(z, z) / sTz
    B_new = B - Bs_outer + zz_outer

    # --- 対称化（丸め誤差対策）である ---
    B_new = 0.5 * (B_new + B_new.T)
    return B_new
