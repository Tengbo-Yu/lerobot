# CUDA_VISIBLE_DEVICES=2 python lerobot/scripts/train.py \
#   --dataset.root=/nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0414 \
#   --dataset.repo_id=JackYuuuu/pick_place_0414 \
#   --policy.type=act \
#   --output_dir=outputs/train/act_0414\
#   --job_name=act_0414 \
#   --policy.device=cuda \
#   --wandb.enable=false 


# CUDA_VISIBLE_DEVICES=4 python lerobot/scripts/train.py \
#   --dataset.root=/nvmessd/ssd_share/tengbo/lerobot/data/pick_place_0414 \
#   --dataset.repo_id=JackYuuuu/pick_place_0414 \
#   --policy.type=diffusion \
#   --output_dir=outputs/train/dp_0414\
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
