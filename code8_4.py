# ソフト閾値関数（soft-thresholding function）の実装である。
# 主に L1 正則化（Lasso）やスパース推定で使用される「近接写像（proximal operator）」の一例。
#
# 背景：
#   φ(x) = C ||x||_1 の近接写像 prox_{φ}(v) は以下の解析解を持つ：
#       prox_{φ}(v) = argmin_x (1/2)||x - v||^2 + C||x||_1
#                    = sign(v) * max(|v| - C, 0)
#
# つまり、各成分ごとに「ソフトなしきい値（soft threshold）」を適用する操作となる。
# これはスパース化（不要な成分をゼロに押しつぶす）に対応しており、
# FISTAやPGMなどの近接勾配法でL1正則化項を扱う際に不可欠である。
#
# 特徴：
#   - |v_i| ≤ C の成分は 0 に置換される。
#   - |v_i| > C の成分は符号を保ったまま |v_i| - C だけ縮小される。
#   - この操作は「soft shrinkage」とも呼ばれる。
#
# 数式で表すと：
#   soft_threshold(v_i, C) =
#       { v_i - C,  if v_i >  C
#       { 0,        if |v_i| ≤ C
#       { v_i + C,  if v_i < -C
#
# 引数：
#   v : np.ndarray または float（入力ベクトル）
#   C : float（しきい値パラメータ；L1正則化強度 λ×α など）
#
# 戻り値：
#   np.ndarray または float（soft-threshold 適用後のベクトル）
#
# 用途例：
#   - FISTA/PGM で L1正則化を含む目的関数を解くとき
#       x_{k+1} = soft_threshold(v, αλ)
#   - Wavelet shrinkage, Compressed sensing などのスパース化処理にも応用される。
import numpy as np


def soft_thresholding(v, C):
    return np.sign(v) * np.maximum(np.abs(v) - C, 0)
