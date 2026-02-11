#!/usr/bin/env bash
# MDLM training on 2 nodes, 8 GPUs per node.
# Adjust time/account as needed.
#SBATCH -A Berzelius-2025-349
#SBATCH -t 1-00:00:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_outs/mdlm-2n-%j.out
#SBATCH --error=slurm_outs/mdlm-2n-%j.err

set -euo pipefail

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /home/x_alfni/.conda/envs/pairwise_env

cd /proj/bedrock/users/x_alfni/denovo_base

# Rendezvous settings
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500
NNODES=${SLURM_NNODES}
NODE_RANK=${SLURM_NODEID}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE}

export MASTER_ADDR MASTER_PORT
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# Optional W&B run name
export WANDB_NAME="mdlm_2node_${SLURM_JOB_ID}"

srun --cpu-bind=none torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  main.py yaml/config_alfred_head.yaml \
  decoder_mdlm.diffusion_config.custom_loss=false \
  batch_size=512
