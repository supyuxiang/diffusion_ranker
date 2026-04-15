
num_candidates=(2 4 6 8 10 12 14 16 18 20)
output_dir=/root/autodl-tmp/fk_search_imagereward

for c in ${num_candidates[@]}; do
python3 /root/diffusion_ranker/fk_search/dit_imagereward.py \
  --output_dir ${output_dir}/num_candidates_${c} \
  --num_samples 50000 \
  --search_batch_size 32 \
  --deno_steps 10 \
  --guidance_scale 1.0 \
  --num_candidates ${c} \
  --tau 0.05 \
  --fk_resample \
  --pick sample
done
