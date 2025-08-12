#!/bin/bash
#SBATCH --job-name=pggcn_scalable
#SBATCH --output=slurm_scalable-large-%j.out
#SBATCH --error=slurm_scalable-large-%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

echo "Starting scalable PGGCN job at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "Node: $(hostname)"

# Check system resources
echo "System resources:"
echo "  CPUs: $(nproc)"
echo "  Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

# Check GPU availability (if any)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "GPU information: nvidia-smi not available (using CPU)"
fi

# Set optimized environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=8

# Monitor memory usage
echo "Initial system memory usage:"
free -h

echo "Starting scalable PGGCN training..."
echo "This version can handle 50+ structures with adaptive memory management"

# Run the scalable script
python /home/lthoma21/BFE-Loss-Function/FINAL-PDBBIND-FILES/experiments/Aug11_2025/test2_pdbbind_scalable_2735.py

echo "Final system memory usage:"
free -h
echo "Job completed at $(date)"
