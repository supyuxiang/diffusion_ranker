#!/usr/bin/env python3
"""
最小示例：PyTorch Module 上的 forward hook / forward pre-hook。

运行: python3 demo/demo_hook.py
"""

from __future__ import annotations

import torch
import torch.nn as nn


def main() -> None:
    # 简单 MLP：2 -> 4 -> 1
    model = nn.Sequential(
        nn.Linear(2, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )

    # 目标：hook 在第一层 Linear 上，观察「输入 / 输出」
    first_linear: nn.Linear = model[0]

    def forward_pre_hook(module, inputs):
        x, = inputs
        print(f"[pre-hook]  {module.__class__.__name__} 输入 shape={tuple(x.shape)}")

    def forward_hook(module, inputs, output):
        print(f"[forward-hook] {module.__class__.__name__} 输出 shape={tuple(output.shape)}")
        # 可原地改输出（一般慎用）；这里只演示可读性
        return None  # 返回 None 表示不替换 output

    h_pre = first_linear.register_forward_pre_hook(forward_pre_hook)
    h_fwd = first_linear.register_forward_hook(forward_hook)

    x = torch.randn(3, 2)  # batch=3
    y = model(x)
    print(f"最终输出 shape={tuple(y.shape)}")

    # 用完可移除（避免多次 forward 重复注册多份时混乱）
    h_pre.remove()
    h_fwd.remove()

    print("\n--- backward hook 示例（可选）---")

    def backward_hook(module, grad_input, grad_output):
        # grad_output[0]: 对本层输出的梯度 dL/d(out)
        g = grad_output[0]
        print(f"[backward-hook] {module.__class__.__name__} grad_output shape={tuple(g.shape)}")
        return None

    lin = model[2]
    h_b = lin.register_full_backward_hook(backward_hook)
    x2 = torch.randn(2, 2, requires_grad=True)
    out = model(x2).sum()
    out.backward()
    h_b.remove()


if __name__ == "__main__":
    main()
