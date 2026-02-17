#!/usr/bin/env bash
# MKB2 LR sweep on MDLM Lance (2 nodes, 8 GPUs per node).
# Sweeps lr_warmup_end x custom_loss.
#SBATCH -A Berzelius-2025-349
#SBATCH -t 0-15:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --array=0-9%2
#SBATCH --output=slurm_outs/sweep-mkb2-%A_%a.out
#SBATCH --error=slurm_outs/sweep-mkb2-%A_%a.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/bedrock/users/x_alfni/conda-envs/BedrockEnv

cd /proj/bedrock/users/x_alfni/denovo_base
set -euo pipefail

# Sweep grid: 5 LR values x 2 custom-loss values = 10 array tasks.
LR_VALUES=(1.0e-4 4.0e-4 8.0e-4 1.0e-3 1.2e-3)
CUSTOM_LOSS_VALUES=(false true)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
N_LR=${#LR_VALUES[@]}
N_CUSTOM=${#CUSTOM_LOSS_VALUES[@]}
TOTAL=$((N_LR * N_CUSTOM))
if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "Invalid task index ${TASK_ID}; expected [0, $((TOTAL - 1))]"
  exit 1
fi

CUSTOM_IDX=$((TASK_ID / N_LR))
LR_IDX=$((TASK_ID % N_LR))
LR_WARMUP_END=${LR_VALUES[$LR_IDX]}
CUSTOM_LOSS=${CUSTOM_LOSS_VALUES[$CUSTOM_IDX]}

# Fixed run settings for this sweep.
PER_GPU_BATCH_SIZE=512
TRAIN_NUM_WORKERS=2
VAL_NUM_WORKERS=2

# Rendezvous settings.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
NNODES=${SLURM_NNODES}
export MASTER_ADDR MASTER_PORT NNODES

# Runtime settings.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0
export WANDB_NAME="sweep_mkb2_lr${LR_WARMUP_END}_cl${CUSTOM_LOSS}_${SLURM_JOB_ID}_${TASK_ID}"
TORCHRUN=/proj/bedrock/users/x_alfni/conda-envs/BedrockEnv/bin/torchrun
export TORCHRUN

echo "[sweep] task=${TASK_ID}/${TOTAL} lr_warmup_end=${LR_WARMUP_END} custom_loss=${CUSTOM_LOSS}"

# Stage dataset on each node local NVMe.
ARRAY_ROOT=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
ARRAY_TASK=${SLURM_ARRAY_TASK_ID:-0}
SRC_DATA=/proj/bedrock/datasets/MKB2
SCRATCH_DATA=/scratch/local/${ARRAY_ROOT}_${ARRAY_TASK}/MKB2
export SRC_DATA SCRATCH_DATA

srun --nodes=$NNODES --ntasks=$NNODES --ntasks-per-node=1 --distribution=block:block bash -c '
set -euo pipefail
mkdir -p "$SCRATCH_DATA"
rsync -a --delete "$SRC_DATA"/lance/ "$SCRATCH_DATA"/lance/
rsync -a "$SRC_DATA"/dictionary.tsv "$SRC_DATA"/masses.tsv "$SRC_DATA"/enumerate_tokens.py "$SCRATCH_DATA"/
echo "[copied] host=$(hostname) -> $SCRATCH_DATA"
'

srun --nodes=$NNODES --ntasks=$NNODES --ntasks-per-node=1 --distribution=block:block \
  --cpu-bind=none --kill-on-bad-exit=1 bash -c '
NPROC=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[launch] host=$(hostname) nodeid=$SLURM_NODEID nproc=$NPROC cvd=$CUDA_VISIBLE_DEVICES"
$TORCHRUN \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC \
  --node_rank=$SLURM_NODEID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main.py yaml/config_mkb2.yaml \
  loader.use_lance=true \
  loader.lance_species_root=$SCRATCH_DATA/lance \
  loader.val_name=mkb_val \
  loader.train_dataset_path=$SCRATCH_DATA \
  loader.val_dataset_path=$SCRATCH_DATA \
  loader.tokenizer_path=$SCRATCH_DATA \
  loader.masses_path=$SCRATCH_DATA \
  loader.dictionary_path=$SCRATCH_DATA/dictionary.tsv \
  loader.num_workers='"$TRAIN_NUM_WORKERS"' \
  loader.val_num_workers='"$VAL_NUM_WORKERS"' \
  decoder_mdlm.diffusion_config.custom_loss='"$CUSTOM_LOSS"' \
  lr_warmup_end='"$LR_WARMUP_END"' \
  batch_size='"$PER_GPU_BATCH_SIZE"'
'
