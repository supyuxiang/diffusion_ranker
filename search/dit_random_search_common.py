"""Shared helpers for ``dit_random_search_ranker.py`` (DiT rollout + repr capture).

不用这个文件了

"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from utils import DitRolloutReprCapture, _prepare_pil_for_save, rollout_dit_with_steps

# Upper bound for ``numpy.random.Generator.integers`` (exclusive high).
CANDIDATE_SEED_HIGH = 2**31


def init_random_samples(
    pipe,
    output_dir: Path,
    num_init: int,
    batch_size: int,
    inference_steps: int,
    device: torch.device,
    seed: int,
    *,
    guidance_scale: float = 1.0,
) -> int:
    """Write ``num_init`` images under ``output_dir``; same layout as ``rollout_dit_with_steps``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return int(
        rollout_dit_with_steps(
            pipe,
            output_dir,
            inference_steps,
            num_init,
            batch_size,
            device,
            seed,
            guidance_scale=guidance_scale,
            repr_capture=None,
        )
    )


@torch.inference_mode()
def rollout_dit_batch_with_repr(
    pipe,
    repr_capture: DitRolloutReprCapture,
    deno_steps: int,
    n_this: int,
    device: torch.device,
    cand_seed: int,
    *,
    guidance_scale: float = 1.0,
    image_size: int = 256,
) -> tuple[list[Image.Image], list[int], dict]:
    """
    One DiT batch with hooks on; returns PIL copies, labels, and repr dict (keys ``s*_*``).

    Pops the last ``repr_capture.records`` entry so long random-search runs do not grow memory.
    Label sampling matches ``rollout_dit_with_steps`` (uniform random classes, independent of
    ``cand_seed``); images differ via ``generator=manual_seed(cand_seed)``.
    """
    num_classes = 1000
    batch_labels_ls = torch.randint(
        0, num_classes, (n_this,), device=device
    ).tolist()
    gen = torch.Generator(device=device).manual_seed(int(cand_seed) % CANDIDATE_SEED_HIGH)

    repr_capture.set_capturing(True)
    repr_capture.begin_batch(batch_labels_ls)
    out = pipe(
        class_labels=batch_labels_ls,
        num_inference_steps=deno_steps,
        guidance_scale=guidance_scale,
        generator=gen,
    )
    repr_capture.end_batch(0, 0, generator_seed=int(cand_seed) % CANDIDATE_SEED_HIGH)
    rec = repr_capture.records.pop()
    repr_dict = rec["repr"]

    images: list[Image.Image] = []
    for img in out.images:
        images.append(_prepare_pil_for_save(img, image_size).copy())

    del out, gen
    return images, batch_labels_ls, repr_dict
