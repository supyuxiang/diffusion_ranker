#!/usr/bin/env python3
"""
DiT Random Search with CLIPScore Verifier（仅预生成图）

须先用 ``search/rollout.py`` 生成 ``candidate_0000`` …（``image_{idx}_{label}.png``）与可选 ``init_1024``。
每轮从 K 个 candidate 池的**同一位置切片**取 ``batch_size`` 张图，算 batch 平均 CLIP 图文余弦相似度，
选**最高**的一批写入 ``output_dir``。**不加载 DiT**。

默认 **search batch_size=32**（每轮从每个池取 32 张比 FID 版更小，可按需改 ``--batch_size``）。

用法::

  python3 search/dit_random_search_clipscore.py \\
      --imagenet_val /root/autodl-tmp/imagenet/val \\
      --output_dir /root/autodl-tmp/eval_dit_rs_clip \\
      --deno_steps 10 --guidance_scale 1.0 \\
      --num_candidates 16 --batch_size 32 --init_random_samples 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from dit_random_search_common import load_imagenet_simple_labels, prompt_for_class

from utils import compute_fid_is_from_dirs, set_device, set_seed

_DEFAULT_HF_CLIP_ROOT = Path(
    os.environ.get(
        "CLIP_HF_CACHE",
        "/root/autodl-tmp/models--openai--clip-vit-large-patch14",
    )
)


def default_candidates_root(guidance_scale: float, deno_steps: int) -> Path:
    return Path(
        f"/root/autodl-tmp/dit_candidates_50k_cfg{guidance_scale}_deno{deno_steps}"
    )


def sorted_rollout_image_paths(directory: Path) -> list[Path]:
    return sorted(
        directory.glob("image_*_*.png"),
        key=lambda p: int(p.stem.split("_")[1]),
    )


def class_id_from_rollout_path(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"expected image_*_*_label.png, got {path.name}")
    return int(parts[2])


def load_images_and_class_ids(paths: list[Path]) -> tuple[list[Image.Image], list[int]]:
    images: list[Image.Image] = []
    cids: list[int] = []
    for p in paths:
        with Image.open(p) as im:
            images.append(im.convert("RGB").copy())
        cids.append(class_id_from_rollout_path(p))
    return images, cids


def build_candidate_path_lists(root: Path, num_candidates: int) -> list[list[Path]]:
    lists: list[list[Path]] = []
    for i in range(num_candidates):
        d = root / f"candidate_{i:04d}"
        if not d.is_dir():
            raise FileNotFoundError(f"missing candidate dir: {d}")
        pl = sorted_rollout_image_paths(d)
        if not pl:
            raise FileNotFoundError(f"no image_*_*.png under {d}")
        lists.append(pl)
    return lists


def resolve_init_paths(
    candidates_root: Path,
    init_dir: Path | None,
    n_init: int,
    candidate_lists: list[list[Path]],
) -> list[Path]:
    if n_init <= 0:
        return []
    if init_dir is not None and init_dir.is_dir():
        pl = sorted_rollout_image_paths(init_dir)
        if len(pl) < n_init:
            raise RuntimeError(f"{init_dir}: need >= {n_init} images, got {len(pl)}")
        return pl[:n_init]
    pl0 = candidate_lists[0]
    if len(pl0) < n_init:
        raise RuntimeError(f"candidate_0000: need >= {n_init} images for init, got {len(pl0)}")
    return pl0[:n_init]


def resolve_hf_clip_snapshot_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path
    ref = path / "refs" / "main"
    if ref.is_file():
        snap_id = ref.read_text(encoding="utf-8").strip()
        cand = path / "snapshots" / snap_id
        if (cand / "config.json").is_file():
            return cand
    raise FileNotFoundError(
        f"无法在 {path} 解析 CLIP 权重（需要 config.json 或 refs/main + snapshots/<hash>）"
    )


def load_hf_clip(model_dir: Path, device: str):
    from transformers import CLIPModel, CLIPProcessor

    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to(device).float().eval()
    return model, processor


def load_open_clip(model_name: str, pretrained: str, device: str):
    import open_clip
    from torchvision import transforms

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).float().eval()
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )
    return model, tokenizer, transform


def mean_batch_clipscore_hf(
    model,
    processor,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    rgb = [img.convert("RGB") for img in images]
    inputs = processor(text=prompts, images=rgb, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        image_features = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_features = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def mean_batch_clipscore_open_clip(
    model,
    tokenizer,
    image_transform,
    device: str,
    images: list[Image.Image],
    class_ids: list[int],
    labels_1k: list[str],
    prompt_template: str,
) -> float:
    prompts = [prompt_for_class(labels_1k, cid, prompt_template) for cid in class_ids]
    text_tokens = tokenizer(prompts).to(device)
    tensors = torch.stack(
        [image_transform(img.convert("RGB")) for img in images],
        dim=0,
    ).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = model.encode_image(tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sim = (image_features * text_features).sum(dim=-1)
    return float(sim.mean().item()) if sim.numel() else float("-inf")


def init_from_precomputed(output_dir: Path, init_paths: list[Path]) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(init_paths):
        with Image.open(p) as im:
            im.convert("RGB").save(output_dir / f"{i:06d}.png")
    return len(init_paths)


@torch.inference_mode()
def run_random_search_clipscore(
    clip_backend: str,
    clip_model,
    clip_aux,
    labels_1k: list[str],
    prompt_template: str,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_candidates: int,
    device: str,
    candidate_lists: list[list[Path]],
    n_init: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cursor = n_init
    img_counter = n_init
    num_rounds = (num_samples - n_init + batch_size - 1) // batch_size

    for c in range(num_candidates):
        if len(candidate_lists[c]) < num_samples:
            raise RuntimeError(
                f"candidate_{c:04d}: need >= {num_samples} images, got {len(candidate_lists[c])}"
            )

    for _ in tqdm(range(num_rounds), desc="random search clipscore"):
        n_this = min(batch_size, num_samples - cursor)
        if n_this <= 0:
            break

        best_score = float("-inf")
        best_batch: list[Image.Image] | None = None

        for c in range(num_candidates):
            sub = candidate_lists[c][cursor : cursor + n_this]
            batch_images, batch_classes = load_images_and_class_ids(sub)
            if clip_backend == "hf":
                score = mean_batch_clipscore_hf(
                    clip_model,
                    clip_aux,
                    device,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            else:
                tokenizer, image_transform = clip_aux
                score = mean_batch_clipscore_open_clip(
                    clip_model,
                    tokenizer,
                    image_transform,
                    device,
                    batch_images,
                    batch_classes,
                    labels_1k,
                    prompt_template,
                )
            if score > best_score:
                best_score = score
                best_batch = batch_images

        assert best_batch is not None
        for img in best_batch:
            if img_counter >= num_samples:
                break
            img.save(output_dir / f"{img_counter:06d}.png")
            img_counter += 1
        cursor += n_this


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--imagenet_val", default="/root/autodl-tmp/imagenet/val")
    p.add_argument("--output_dir", default="/root/autodl-tmp/eval_dit_random_search_clipscore")
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="每轮每个 candidate 池切片长度（search batch，默认 32）",
    )
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--init_random_samples", type=int, default=32)
    p.add_argument("--deno_steps", type=int, default=20, help="与 rollout 目录名一致")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance_scale", type=float, default=1.0)
    p.add_argument("--metrics_batch_size", type=int, default=64)
    p.add_argument(
        "--candidates_root",
        default="",
        help="预生成根目录；空则用 dit_candidates_50k_cfg{gs}_deno{steps}",
    )
    p.add_argument(
        "--init_dir",
        default="",
        help="init 目录；空则尝试 candidates_root/init_1024，否则 candidate_0000 前 N 张",
    )
    p.add_argument(
        "--prompt_template",
        default="a high-quality picture of a {name}",
        help="{name}、{id} 见 prompt_for_class",
    )
    p.add_argument("--clip_backend", choices=("hf", "open_clip"), default="hf")
    p.add_argument("--clip_model_path", type=str, default=str(_DEFAULT_HF_CLIP_ROOT))
    p.add_argument("--clip_model", default="ViT-B-32")
    p.add_argument("--clip_pretrained", default="openai")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imagenet_val = Path(args.imagenet_val).expanduser().resolve()
    if not imagenet_val.is_dir():
        raise FileNotFoundError(f"--imagenet_val not a directory: {imagenet_val}")

    set_seed(args.seed)
    device = set_device()
    dstr = "cuda" if device.type == "cuda" else "cpu"

    labels_1k = load_imagenet_simple_labels()
    clip_dir_resolved: Path | None = None
    if args.clip_backend == "hf":
        clip_dir_resolved = resolve_hf_clip_snapshot_dir(Path(args.clip_model_path))
        print(f"load HF CLIP: {clip_dir_resolved}")
        clip_model, clip_aux = load_hf_clip(clip_dir_resolved, dstr)
    else:
        print(f"load OpenCLIP {args.clip_model}/{args.clip_pretrained}")
        clip_model, clip_tok, clip_tf = load_open_clip(
            args.clip_model, args.clip_pretrained, dstr
        )
        clip_aux = (clip_tok, clip_tf)

    cr = (
        Path(args.candidates_root).expanduser().resolve()
        if args.candidates_root
        else default_candidates_root(args.guidance_scale, args.deno_steps)
    )
    if not cr.is_dir():
        raise FileNotFoundError(
            f"candidates root missing: {cr}\nRun search/rollout.py per candidate_idx."
        )

    init_dir_opt = Path(args.init_dir).expanduser().resolve() if args.init_dir else None
    auto_init = cr / "init_1024"
    if init_dir_opt is None and auto_init.is_dir() and sorted_rollout_image_paths(auto_init):
        init_dir_opt = auto_init

    lists = build_candidate_path_lists(cr, args.num_candidates)

    n_init = 0
    if args.init_random_samples > 0:
        init_paths = resolve_init_paths(cr, init_dir_opt, args.init_random_samples, lists)
        print(f"init from disk ({len(init_paths)} images) -> {output_dir}")
        n_init = init_from_precomputed(output_dir, init_paths)

    print(
        f"random search (clipscore): root={cr}, "
        f"num_candidates={args.num_candidates}, batch_size={args.batch_size}"
    )
    run_random_search_clipscore(
        args.clip_backend,
        clip_model,
        clip_aux,
        labels_1k,
        args.prompt_template,
        output_dir,
        args.num_samples,
        args.batch_size,
        args.num_candidates,
        dstr,
        lists,
        n_init=n_init,
    )

    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    resolved_root = (
        str(Path(args.candidates_root).expanduser().resolve())
        if args.candidates_root
        else str(default_candidates_root(args.guidance_scale, args.deno_steps))
    )
    cfg = {
        "verifier": "clipscore",
        "clip_backend": args.clip_backend,
        "init_random_samples": args.init_random_samples,
        "deno_steps": args.deno_steps,
        "guidance_scale": args.guidance_scale,
        "num_candidates": args.num_candidates,
        "batch_size": args.batch_size,
        "prompt_template": args.prompt_template,
        "candidates_root": resolved_root,
        "init_dir": str(Path(args.init_dir).expanduser().resolve()) if args.init_dir else "",
    }
    if args.clip_backend == "hf" and clip_dir_resolved is not None:
        cfg["clip_model_path"] = args.clip_model_path
        cfg["clip_model_path_resolved"] = str(clip_dir_resolved)
    else:
        cfg["clip_model"] = args.clip_model
        cfg["clip_pretrained"] = args.clip_pretrained

    log_path = output_dir / "config.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"config saved to {log_path}")

    print("\nfinal FID / IS (torch_fidelity)")
    metrics = compute_fid_is_from_dirs(
        output_dir,
        imagenet_val,
        device,
        batch_size=args.metrics_batch_size,
    )
    print(f"   FID: {metrics['FID']:.2f}, IS: {metrics['IS']:.4f} ± {metrics['IS_STD']:.4f}")
    with open(output_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)


if __name__ == "__main__":
    main()
