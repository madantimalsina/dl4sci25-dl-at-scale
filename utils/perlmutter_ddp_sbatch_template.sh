#!/bin/bash
#SBATCH -C gpu
#SBATCH -A {account}
#SBATCH -q {queue}
#SBATCH -N {num_nodes}
#SBATCH --ntasks-per-node {gpus_per_node}
#SBATCH --cpus-per-task {cpus_per_task}
#SBATCH --gpus-per-node {gpus_per_node}
#SBATCH --time {time_limit}
#SBATCH --image {image}
#SBATCH --reservation={reservation}
#SBATCH --module=gpu,nccl-plugin
#SBATCH -J {job_name}
#SBATCH -o {log_path}/%x-%j.out
#SBATCH --exclusive

# --- user mounts ---
DATADIR={data_dir}
LOGDIR={log_path}
mkdir -p "${LOGDIR}"

# --- repo checkout (reproducible) ---
RUN_DIR={run_dir}
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"
if [ ! -d dl4sci25-dl-at-scale ]; then
  git clone {repo}
fi
cd dl4sci25-dl-at-scale
git fetch origin
git checkout {branch}
git reset --hard origin/{branch}
COMMIT=$(git rev-parse --short HEAD)
echo "Using commit: ${COMMIT}"

# --- light env (NCCL handled by module=gpu,nccl-plugin) ---
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS={omp_threads}
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1

# torchrun rendezvous
export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT=${MASTER_PORT:-29500}

# ClearML labels (rank-0 wrapper will init)
export CLEARML_PROJECT={clearml_project}
export CLEARML_TASK_NAME="{job_name}-\${SLURM_JOB_NUM_NODES}x\${SLURM_GPUS_ON_NODE:-{gpus_per_node}}@\${COMMIT}"

APP=utils/ddp_multinode_clearml.py  # rank-0 ClearML init; then exec train.py

set -x
srun -u \
  --ntasks-per-node={gpus_per_node} \
  --gpus-per-task=1 \
  --cpu-bind=cores \
  --accel-bind=g \
  --kill-on-bad-exit=1 \
  shifter -V "${DATADIR}:/data" -V "${LOGDIR}:/logs" \
    bash -lc "
      echo 'Inside container on:' \$(hostname)
      python -m torch.distributed.run \
        --nnodes \${SLURM_JOB_NUM_NODES} \
        --nproc_per_node {gpus_per_node} \
        --rdzv_backend c10d \
        --rdzv_endpoint \${MASTER_ADDR}:\${MASTER_PORT} \
        \${APP} {train_args}
    "
