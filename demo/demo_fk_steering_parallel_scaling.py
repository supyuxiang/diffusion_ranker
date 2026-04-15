#!/usr/bin/env python3
r"""
FK Steering（Feynman–Kac Steering）与 **并行 scaling** —— 极简演示
================================================================

背景（直觉）
-----------

扩散生成可看成一条随机轨迹 :math:`x_{0:T}`。若在中间时刻 :math:`t` 上有一个**奖励/势函数**
:math:`r(x_t)`（越大越好），我们希望最终样本更常落在高奖励区域，但不想改训练好的 score 网络。

**FK Steering** 的做法是同时跑 **N 条并行轨迹（particles）**，在若干 checkpoint 上：

1. 用当前状态算权重（与 :math:`r` 或累积势相关，常见形式如 importance weight 的乘积）；
2. **多项式重采样（resample）**：按权重从 N 条里**有放回**抽取 N 条，让“好粒子”被复制、差粒子被淘汰；
3. 继续从被选中粒子的状态往后扩散。

这与序贯蒙特卡洛（SMC）/ 粒子滤波同族；名字来自 Feynman–Kac 公式把“带势的路径测度”与粒子近似联系起来。

**并行 scaling** 指：计算上这 N 条前向在 GPU 上可 **batch 并行**；统计上 **N 越大**，同样步数下逼近高奖励分布越有效（方差下降、best-of-N 类效果增强），但每步要付 **O(N)** 的奖励评估与重采样开销。

本脚本 **不调用真实 DiT/UNet**，只用 1D 随机游走 + 显式高斯奖励，演示 **resample 如何把粒子质量推向目标**。

运行::

    python3 demo/demo_fk_steering_parallel_scaling.py
"""

from __future__ import annotations

import math
import random

import torch


def gaussian_reward(x: torch.Tensor, target: float = 0.0, sigma: float = 0.5) -> torch.Tensor:
    """标量奖励：离 target 越近越大。shape (N,) -> (N,)"""
    return torch.exp(-((x - target) ** 2) / (2 * sigma**2))


def fk_steering_step(
    x: torch.Tensor,
    *,
    target: float,
    noise_std: float,
    resample: bool,
    ess_threshold: float = 0.5,
) -> tuple[torch.Tensor, float]:
    """
    一步：随机游走 +（可选）按奖励权重重采样。

    - x: (N,) 当前粒子位置
    - 若 resample：用 w ∝ r(x) 做多项式重采样；可选 ESS 触发（简化版：每步都 resample 便于教学）
    返回 (新 x, 本步平均奖励)
    """
    n = x.shape[0]
    # 前向扩散式微扰（玩具 dynamics）
    x = x + noise_std * torch.randn_like(x)
    r = gaussian_reward(x, target=target)
    mean_r = float(r.mean().item())

    if not resample:
        return x, mean_r

    w = r / (r.sum() + 1e-12)
    # 有效样本量 ESS = 1 / sum(w^2)；小说明权重塌缩，更该 resample
    ess = float(1.0 / (w**2).sum().item())
    if ess < ess_threshold * n:
        pass  # 已决定每步都 resample；真实系统可仅在 ESS 低时 resample

    idx = torch.multinomial(w, num_samples=n, replacement=True)
    x = x[idx].clone()
    return x, mean_r


def run_experiment(
    *,
    num_particles: int,
    num_steps: int,
    use_fk: bool,
    seed: int,
    target: float = 0.0,
) -> dict[str, float]:
    torch.manual_seed(seed)
    random.seed(seed)
    x = torch.randn(num_particles) * 2.0  # 分散初值
    traj_mean_r: list[float] = []
    for _ in range(num_steps):
        x, mr = fk_steering_step(
            x,
            target=target,
            noise_std=0.35,
            resample=use_fk,
        )
        traj_mean_r.append(mr)
    final_dist = float(torch.abs(x - target).mean().item())
    return {
        "final_mean_abs_err": final_dist,
        "mean_reward_last_step": traj_mean_r[-1] if traj_mean_r else 0.0,
        "mean_reward_auc": float(sum(traj_mean_r) / max(len(traj_mean_r), 1)),
    }


def main() -> None:
    print(__doc__.split("====")[0].strip())
    print("\n--- 对比：同一随机种子、同一初值分布 ---\n")

    Ns = [4, 16, 64]
    steps = 30
    seed0 = 42

    for N in Ns:
        a = run_experiment(num_particles=N, num_steps=steps, use_fk=False, seed=seed0)
        b = run_experiment(num_particles=N, num_steps=steps, use_fk=True, seed=seed0)
        print(f"N = {N:3d}  particles, T = {steps}")
        print(f"  无 FK（仅并行独立噪声）  末步平均 |x-target| = {a['final_mean_abs_err']:.4f}")
        print(f"  有 FK Steering          末步平均 |x-target| = {b['final_mean_abs_err']:.4f}")
        print(f"  （末步平均奖励 ↑ 越好） 无FK={a['mean_reward_last_step']:.4f}  有FK={b['mean_reward_last_step']:.4f}")
        print()

    print(
        "解读：\n"
        "  • **并行 N**：同一时刻算 N 个粒子的奖励与权重，可 batch 向量化（真实扩散即 batch 多轨迹）。\n"
        "  • **Scaling**：N 增大通常降低估计方差、提高找到高奖励区域的概率（与 best-of-N 相关但每步有交互）。\n"
        "  • **代价**：每 resample 步需要 O(N) 奖励调用 + 多项式采样；论文中常与步数、调度 trade-off。\n"
    )
    print(
        "参考文献方向：\n"
        "  • “Fk-Diffusion-Steering” 等开源实现；\n"
        "  • MLR / arXiv: inference-time scaling & steering of diffusion（Feynman–Kac 视角）。\n"
    )


if __name__ == "__main__":
    main()
