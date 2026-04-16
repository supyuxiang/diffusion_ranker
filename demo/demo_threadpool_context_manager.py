#!/usr/bin/env python3
"""
ThreadPoolExecutor demo:
1) `with` context manager lifecycle
2) `as_completed` completion order vs submit order
3) ordered collection and retry extension

Run:
  python3 demo/demo_threadpool_context_manager.py --num_tasks 12 --max_workers 4
"""
from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def fake_save_task(task_idx: int, out_path: Path, fail_once: bool = False) -> dict:
    """
    Simulate "save image to disk":
    - variable runtime
    - optional one-time failure
    """
    # Simulate variable I/O latency
    delay = random.uniform(0.02, 0.18)
    time.sleep(delay)

    # Optional controlled failure (for retry demo)
    if fail_once and task_idx % 5 == 0 and not out_path.with_suffix(".retry_ok").exists():
        # mark that this task already failed once
        out_path.with_suffix(".retry_ok").write_text("1", encoding="utf-8")
        raise RuntimeError(f"task {task_idx} simulated transient failure")

    # Write a small payload
    out_path.write_text(f"task={task_idx}, delay={delay:.3f}s\n", encoding="utf-8")
    return {"task_idx": task_idx, "delay": delay, "path": str(out_path)}


def demo_as_completed(out_dir: Path, num_tasks: int, max_workers: int) -> None:
    print("\n=== Demo A: as_completed (completion-order loop) ===")
    submit_order: list[int] = []
    completion_order: list[int] = []

    # `with` ensures executor.shutdown(wait=True) on exit
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for idx in range(num_tasks):
            submit_order.append(idx)
            out_path = out_dir / f"image_{idx:05d}.txt"
            fut = ex.submit(fake_save_task, idx, out_path, False)
            futures.append(fut)

        for fut in as_completed(futures):
            # .result() re-raises exceptions from worker thread
            info = fut.result()
            completion_order.append(info["task_idx"])

    print(f"submit order     : {submit_order}")
    print(f"completion order : {completion_order}")
    print("note: completion order can differ, but each task still writes to its own path.")


def demo_ordered_collect(out_dir: Path, num_tasks: int, max_workers: int) -> None:
    print("\n=== Demo B: keep submit-order when collecting results ===")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for idx in range(num_tasks):
            out_path = out_dir / f"ordered_{idx:05d}.txt"
            futures.append(ex.submit(fake_save_task, idx, out_path, False))

        # Iterate futures list directly => submit order
        ordered = [f.result()["task_idx"] for f in futures]
    print(f"collected order  : {ordered}")


def demo_retry(out_dir: Path, num_tasks: int, max_workers: int, max_retries: int) -> None:
    print("\n=== Demo C: extension - retry transient failures ===")
    tasks = list(range(num_tasks))
    success: dict[int, str] = {}
    attempt = 0

    while tasks and attempt <= max_retries:
        attempt += 1
        print(f"attempt {attempt}, pending={len(tasks)}")
        failed_next: list[int] = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_idx = {
                ex.submit(fake_save_task, idx, out_dir / f"retry_{idx:05d}.txt", True): idx
                for idx in tasks
            }
            for fut in as_completed(fut_to_idx):
                idx = fut_to_idx[fut]
                try:
                    info = fut.result()
                    success[idx] = info["path"]
                except Exception as e:
                    print(f"  task {idx} failed: {e}")
                    failed_next.append(idx)

        tasks = failed_next

    print(f"success={len(success)}, unresolved={len(tasks)}")
    if tasks:
        print(f"unresolved tasks: {tasks}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--max_workers", type=int, default=4)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--out_dir", type=str, default="/tmp/threadpool_demo")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("ThreadPoolExecutor context-manager demo")
    print(f"out_dir={out_dir}")
    print(f"num_tasks={args.num_tasks}, max_workers={args.max_workers}")

    demo_as_completed(out_dir, args.num_tasks, args.max_workers)
    demo_ordered_collect(out_dir, args.num_tasks, args.max_workers)
    demo_retry(out_dir, args.num_tasks, args.max_workers, args.max_retries)

    print("\nDone.")


if __name__ == "__main__":
    main()
