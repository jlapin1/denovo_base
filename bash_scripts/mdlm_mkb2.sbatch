#!/usr/bin/env bash
# MDLM Lance training on MKB2 (2 nodes, 8 GPUs per node).
# Uses local scratch staging + Lance map-style loader.
#SBATCH -A Berzelius-2025-349
#SBATCH -t 0-72:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --output=slurm_outs/mdlm-mkb2-%j.out
#SBATCH --error=slurm_outs/mdlm-mkb2-%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/bedrock/users/x_alfni/conda-envs/BedrockEnv

cd /proj/bedrock/users/x_alfni/denovo_base
set -euo pipefail

# Rendezvous settings.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
NNODES=${SLURM_NNODES}
export MASTER_ADDR MASTER_PORT NNODES

# Runtime settings.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0
export WANDB_NAME="mdlm_mkb2_2n8gpu_${SLURM_JOB_ID}"
TORCHRUN=/proj/bedrock/users/x_alfni/conda-envs/BedrockEnv/bin/torchrun
export TORCHRUN

# Stage dataset on each node local NVMe.
SRC_DATA=/proj/bedrock/datasets/MKB2
SCRATCH_DATA=/scratch/local/$SLURM_JOB_ID/MKB2
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
  loader.num_workers=2 \
  loader.val_num_workers=2 \
  decoder_mdlm.diffusion_config.custom_loss=false \
  batch_size=512'
