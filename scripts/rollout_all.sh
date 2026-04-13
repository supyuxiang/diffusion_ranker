

deno_steps_ls=(10 25 50 100)
idx_ls=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)


for deno_step in ${deno_steps_ls[@]}; do
for idx in ${idx_ls[@]}; do
CUDA_VISIBLE_DEVICES=0 python3 /root/diffusion_ranker/search/rollout.py \
  --candidate_idx ${idx} \
  --deno_steps ${deno_step} \
  --guidance_scale 1.0 \
  --num_samples 50000 \
  --batch_size 512 \
  --dtype bf16
done
done


# compute metrics : fid, is ,clipscore
# TODO
