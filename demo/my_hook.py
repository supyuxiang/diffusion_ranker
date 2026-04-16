import torch
import torch.nn as nn
from pathlib import Path




# model = nn.Sequential(
#     nn.Linear(10, 20),
#     nn.ReLU(),
#     nn.Linear(20, 10),
# )


# repr_ls = []

# def hook_fn(module,input,output):
#     repr_ls.append(output.detach().cpu())


# handle_ls = []

# for m in model.modules():
#     handle_ls.append(m.register_forward_hook(hook_fn))

# inputs = torch.randn(5, 10)
# model(inputs)

# print(repr_ls)
# print(len(repr_ls))

# for handle in handle_ls:
#     handle.remove()

# print(repr_ls[0])




_target_layer_ratio_ls = [0.1,0.3,0.6,0.9,1.0]

_target_step_ratio_ls = [0.0,0.25,0.5,0.75,1.0]







def demo_unet_ddpm_hook():
    """
    UNet-based DDPM hook demo:
    - load local DDPM pipeline
    - register hooks on selected ResnetBlock2D modules
    - run a few denoising steps
    - print captured feature shapes
    """
    from diffusers import DDPMPipeline

    model_dir = Path("/root/autodl-tmp/google_ddpm-celebahq-256")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = DDPMPipeline.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 1) choose hook modules from UNet
    resnet_blocks = [
        (name, mod)
        for name, mod in pipe.unet.named_modules()
        if mod.__class__.__name__ == "ResnetBlock2D"
    ]
    print('--------------------------------')
    print(resnet_blocks)
    print('--------------------------------')
    if not resnet_blocks:
        print("[ddpm hook demo] no ResnetBlock2D found")
        return
    print(f"[ddpm hook demo] total ResnetBlock2D blocks: {len(resnet_blocks)}")

    pick_idx = sorted(set([0, len(resnet_blocks) // 2, len(resnet_blocks) - 1]))
    picked = [resnet_blocks[i] for i in pick_idx]
    print("[ddpm hook demo] picked blocks:")
    for i, (n, _) in zip(pick_idx, picked):
        print(f"  - idx={i:02d} name={n}")

    # 2) register hooks
    feature_store: dict[str, list[torch.Tensor]] = {name: [] for name, _ in picked}

    def _make_hook(name):
        def _hook(_module, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t):
                feature_store[name].append(t.detach().float().cpu())

        return _hook

    handles = [m.register_forward_hook(_make_hook(name)) for name, m in picked]

    # 3) run a tiny DDPM denoise loop
    batch_size = 2
    num_steps = 5
    generator = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(
        (
            batch_size,
            pipe.unet.config.in_channels,
            pipe.unet.config.sample_size,
            pipe.unet.config.sample_size,
        ),
        generator=generator,
        device=device,
        dtype=next(pipe.unet.parameters()).dtype,
    )
    pipe.scheduler.set_timesteps(num_steps, device=device)
    for t in pipe.scheduler.timesteps:
        eps = pipe.unet(x, t).sample
        x = pipe.scheduler.step(eps, t, x, generator=generator).prev_sample

    # 4) remove hooks and summarize
    for h in handles:
        h.remove()

    print("[ddpm hook demo] captured feature summary:")
    for name, feats in feature_store.items():
        if not feats:
            print(f"  - {name}: 0 captures")
            continue
        last_shape = tuple(feats[-1].shape)
        print(f"  - {name}: captures={len(feats)} last_shape={last_shape}")


print("\n===== UNet-based DDPM hook demo =====")


if __name__ == '__main__':
    demo_unet_ddpm_hook()









