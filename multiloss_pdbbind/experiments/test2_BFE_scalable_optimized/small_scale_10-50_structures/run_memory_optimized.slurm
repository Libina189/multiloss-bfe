#!/bin/bash
#SBATCH --job-name=pggcn_test_memory_opt
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

echo "Starting job at $(date)"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"

# Check GPU availability (if any)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU information:"
    nvidia-smi
else
    echo "GPU information: nvidia-smi not available"
fi

# Set memory-efficient environment variables
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the optimized script
python /home/lthoma21/BFE-Loss-Function/FINAL-PDBBIND-FILES/test2_pdbbind_10_structures.py

echo "Job completed at $(date)"
