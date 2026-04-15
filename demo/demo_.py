from utils import rollout_dit_perclass32_and_save_repr

tensor = rollout_dit_perclass32_and_save_repr(
    layer_ratio=0.6,
    step_ratio=0.25,  # 对应 deno=10 时 s2
    image_output_dir="/root/autodl-tmp/dit_perclass32_s2_r0.6/images",
    repr_output_pt="/root/autodl-tmp/dit_perclass32_s2_r0.6/mean/s2_r0.6.pt",
    model_name="dit-xl-2-256",
    num_inference_steps=10,
    guidance_scale=1.0,
    batch_size=256,
    seed=42,
    dtype="bf16",
    pooling_mode="mean",
)
print(tensor.shape)  # 期望 (32000, hidden_dim)