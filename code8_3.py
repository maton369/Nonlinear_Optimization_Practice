"""
コード8.3 ■ FISTA(アルゴリズム8.4) のコード（説明コメント付き）
"""

import numpy as np


def FISTA(
    obj_f,  # f(x): 滑らかな目的関数（∇f が L-リプシッツ連続）
    nab_f,  # ∇f(x): f の勾配を返す関数
    phi,  # φ(x): 非滑らかな凸関数（例：L1正則化、指示関数など）
    prox,  # prox_{αC}(·): 近接写像（例：soft-threshold、集合への射影）
    C,  # 近接のスケール（例：φ(x)=λ||x||_1 の λ）
    x_k,  # 初期点
    alpha,  # ステップ幅（通常は 0 < α ≤ 1/L を推奨；L は ∇f のリプシッツ定数）
    max_iter=10000,  # 最大反復回数
    eps=1.0e-6,  # 収束判定閾値（||x_k - x_{k-1}|| ≤ eps）
):
    """
    FISTA（Fast Iterative Shrinkage-Thresholding Algorithm）の基本実装である。
    合成最適化
        最小化  F(x) = f(x) + φ(x)
    を、Nesterov型の加速（モメンタム）を付与した近接勾配法で解く。

    ・f は滑らか（∇f が L-リプシッツ連続）、
     φ は一般に凸・非滑らかを仮定する（例：L1、TV、指示関数など）。
    ・収束速度は O(1/k^2)（目的値に関して；標準PGMの O(1/k) より高速）。
    ・ステップ幅 α は概ね 1/L を用いる（L が未知なら別途バックトラッキング版を用意するのが定石）。
    """

    # y_k は「加速後の評価点」（モメンタムを乗せる側）、x_k は更新後の主列である。
    y_k = x_k.copy()  # 初期の加速点を現在点に合わせる
    tau_k = 1.0  # 加速パラメータ t_k（FISTA の標準初期値は 1）

    for k in range(max_iter):
        # --- 退避と勾配計算 ---
        # x_old : 直前の主列（モメンタム計算に使用）
        # nab_f_k : y_k における ∇f を評価（FISTA は y_k を勾配評価点に用いる）
        x_old = x_k
        nab_f_k = nab_f(y_k)

        # --- 近接勾配ステップ（主列の更新）---
        # x_k = prox_{αC}( y_k - α ∇f(y_k) )
        # ここで C は φ のスケール。φ(x)=λ||x||_1 なら prox は成分毎の soft-threshold になる。
        x_k = prox(y_k - alpha * nab_f_k, alpha * C)

        # --- 加速パラメータ更新 ---
        # tau_{k+1} = (1 + sqrt(1 + 4 tau_k^2)) / 2
        # Nesterov の解析に由来する更新式（O(1/k^2) を実現）
        tau_new = (1.0 + np.sqrt(1.0 + 4.0 * tau_k**2)) / 2.0

        # --- モメンタム（加速点）の更新 ---
        # y_{k+1} = x_k + ((tau_k - 1)/tau_{k+1}) * (x_k - x_{k-1})
        # 勾配は y 側で評価しつつ、主列 x の差分で加速する
        y_k = x_k + ((tau_k - 1.0) / tau_new) * (x_k - x_old)

        # --- 収束判定 ---
        # ここでは「主列の変化量」が十分小さいかで判断（他に ||G|| や目的値減少でもよい）
        if np.linalg.norm(x_k - x_old) <= eps:
            break

        # 次ステップへパラメータを送る
        tau_k = tau_new

    # 結果の簡易ログ表示（学習・検証用）
    print(f"FISTA: 反復回数{k+1:d}, 最適値{obj_f(x_k) + phi(x_k):.5e}")
    return x_k
