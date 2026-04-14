





@torch.inference_mode()
def eval_epoch(
    model: torch.nn.Module,
    pt_obj: dict,
    selected_keys: list[str],
    image_root: Path,
    device: torch.device,
    reward_by_name: dict[str, float] | None,
    teacher,
    labels: list[str],
    prompt_template: str,
    ir_device: str,
    dataset: RankerReprGroupDataset,
    args: argparse.Namespace,
    ranker_type: str,
    max_groups: int = 32,
    epoch: int = 0,
) -> dict[str, float]:
    model.eval()
    dataset.set_epoch(epoch)
    sum_tot = sum_mse = sum_kl = 0.0
    n_steps = 0
    eval_rng = random.Random(args.seed + 17_971 + epoch)
    for tok_cpu, t_scores_ir in iter_training_groups(
        dataset,
        pt_obj,
        selected_keys,
        image_root,
        reward_by_name,
        teacher,
        labels,
        prompt_template,
        ir_device,
        max_groups,
        shuffle_rng=eval_rng,
        pool_mode=args.pool_mode,
    ):
        tok = tok_cpu.to(device, non_blocking=True)
        t_scores = t_scores_ir.to(device)
        if ranker_type == "listwise":
            logits = model(tok.unsqueeze(0)).squeeze(0)
        else:
            logits = model(tok)
        loss, mse_v, kl_v = mse_kl_combined_loss(
            logits, t_scores, args.listwise_tau, args.alpha
        )
        sum_tot += float(loss)
        sum_mse += float(mse_v)
        sum_kl += float(kl_v)
        n_steps += 1
        if n_steps >= max_groups:
            break
    if n_steps == 0:
        return {"loss": 0.0, "mse": 0.0, "kl": 0.0}
    return {
        "loss": sum_tot / n_steps,
        "mse": sum_mse / n_steps,
        "kl": sum_kl / n_steps,
    }
