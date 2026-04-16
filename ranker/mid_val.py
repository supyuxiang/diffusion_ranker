#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path("/root/diffusion_ranker/class-condition/val.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()

