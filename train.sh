# CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/train.py \
#   --dataset.root=/nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0406 \
#   --dataset.repo_id=JackYuuuu/pick_place_0406 \
#   --policy.type=act \
#   --output_dir=outputs/train/act_0406\
#   --job_name=act_0406 \
#   --policy.device=cuda \
#   --wandb.enable=false 


# CUDA_VISIBLE_DEVICES=4 python lerobot/scripts/train.py \
#   --dataset.root=/nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0414 \
#   --dataset.repo_id=JackYuuuu/pick_place_0414 \
#   --policy.type=diffusion \
#   --output_dir=outputs/train/debug\
#   --job_name=train/dp_0414 \
#   --policy.device=cuda \
#   --wandb.enable=false

# /nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0414

CUDA_VISIBLE_DEVICES=6 python lerobot/scripts/train.py \
  --dataset.root=/nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0414 \
  --dataset.repo_id=JackYuuuu/pick_place_0414 \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_0414\
  --job_name=train/pi0_0414 \
  --policy.device=cuda \
  --wandb.enable=false

# CUDA_VISIBLE_DEVICES=3 xvfb-run -a python lerobot/scripts/train.py \
#   --dataset.repo_id=lerobot/aloha_sim_insertion_human_image \
#   --policy.type=act \
#   --output_dir=outputs/train/act_sim\
#   --job_name=act_sim \
#   --policy.device=cuda \
#   --wandb.enable=true \
  # --eval_freq=20 \
  # --env.type=aloha