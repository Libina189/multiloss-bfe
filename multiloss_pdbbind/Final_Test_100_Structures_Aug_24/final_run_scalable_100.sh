#!/bin/bash
# Set GPU device (default to GPU 0)
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

# Get timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create logs directory
mkdir -p logs_aug24

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
# export TF_CPP_MIN_LOG_LEVEL=2
# export TF_ENABLE_ONEDNN_OPTS=0
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export OMP_NUM_THREADS=8

# Monitor memory usage
echo "Initial system memory usage:"
free -h

#  Run the training script
echo "Starting training..."
echo "Logs saved to:"
echo "  Output: logs_aug24/gpu_training_${TIMESTAMP}_100.out"
echo "  Errors: logs_aug24/gpu_training_${TIMESTAMP}_100.err"
echo ""

# Run the scalable script
python3 /home/exouser/multiloss-bfe/multiloss_pdbbind/final_test_with_sweep_100.py > logs_aug24/gpu_training_${TIMESTAMP}_100.out 2> logs_aug24/gpu_training_${TIMESTAMP}_100.err

# Check results
if [ $? -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed! Check error logs:"
    echo "  logs_aug24/gpu_training_${TIMESTAMP}_100.err"
fi


echo "Output files: logs_aug24/gpu_training_${TIMESTAMP}_100.out"
echo "Error files: logs_aug24/gpu_training_${TIMESTAMP}_100.err"

echo "Final system memory usage:"
free -h
echo "Job completed at $(date)"
