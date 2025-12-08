#!/bin/bash
###############################################################################
# Lumos5G Training Comparison Experiment
#
# Compares three training approaches:
#   1. Centralized - Single model on full dataset
#   2. Federated   - Collaborative training across clients
#   3. Local-Only  - Each client trains independently
###############################################################################

set -e  # Exit on error

# Configuration
NUM_CLIENTS=5
NUM_ROUNDS=10
EPOCHS=20
EPOCHS_PER_ROUND=2
BATCH_SIZE=256
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="comparison_exp_${TIMESTAMP}"
EXP_DIR="experiments/${EXP_NAME}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "LUMOS5G TRAINING COMPARISON EXPERIMENT"
echo "========================================================================"
echo "Experiment: ${EXP_NAME}"
echo "Clients: ${NUM_CLIENTS}"
echo "FL Rounds: ${NUM_ROUNDS}"
echo "Local Epochs: ${EPOCHS}"
echo "========================================================================"
echo ""

# Create experiment directories
mkdir -p "${EXP_DIR}/logs"
mkdir -p "${EXP_DIR}/models"
mkdir -p "${EXP_DIR}/plots"

###############################################################################
# Step 0: Prerequisites
###############################################################################

echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check for preprocessors
if [ ! -f "federated_data/scaler.pkl" ]; then
    echo -e "${YELLOW}⚠️  Preprocessors not found. Creating them now...${NC}"
    python create_schema_based_preprocessors.py --output_dir federated_data \
        > "${EXP_DIR}/logs/00_preprocessors.log" 2>&1
    echo -e "${GREEN}✓ Preprocessors created${NC}"
fi

# Check for client data splits
if [ ! -f "federated_data/site-1.csv" ]; then
    echo -e "${RED}❌ Client data splits not found${NC}"
    echo "Please run: python split_train_federated.py --data_path train.csv --num_clients ${NUM_CLIENTS}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites satisfied${NC}"
echo ""

###############################################################################
# Step 1: Centralized Training
###############################################################################

echo "========================================================================"
echo "SCENARIO 1: CENTRALIZED TRAINING"
echo "========================================================================"
echo "Training single model on full dataset..."
echo ""

START_TIME=$(date +%s)

python train.py \
    --data_path train.csv \
    --output_dir "${EXP_DIR}/models/centralized" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    > "${EXP_DIR}/logs/01_centralized.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Completed: Centralized Training (${DURATION}s)${NC}"
else
    echo -e "${RED}✗ Failed: Centralized Training${NC}"
    echo "Check log: ${EXP_DIR}/logs/01_centralized.log"
fi
echo ""

###############################################################################
# Step 2: Federated Learning
###############################################################################

echo "========================================================================"
echo "SCENARIO 2: FEDERATED LEARNING"
echo "========================================================================"
echo "Training across ${NUM_CLIENTS} clients with ${NUM_ROUNDS} rounds..."
echo ""

START_TIME=$(date +%s)

python job.py \
    --n_clients ${NUM_CLIENTS} \
    --num_rounds ${NUM_ROUNDS} \
    --epochs_per_round ${EPOCHS_PER_ROUND} \
    --data_dir federated_data \
    --job_name "federated_${EXP_NAME}" \
    --output_dir "${EXP_DIR}/federated_training" \
    > "${EXP_DIR}/logs/02_federated.log" 2>&1

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Completed: Federated Learning (${DURATION}s)${NC}"
    
    # Copy federated results to experiment directory
    if [ -d "${EXP_DIR}/federated_training" ]; then
        mkdir -p "${EXP_DIR}/models/federated"
        cp "${EXP_DIR}/federated_training/federated_${EXP_NAME}/server/simulate_job/app_server/FL_global_model.pt" "${EXP_DIR}/models/federated/."
        echo "  Copied results to ${EXP_DIR}/models/"
    fi
else
    echo -e "${RED}✗ Failed: Federated Learning${NC}"
    echo "Check log: ${EXP_DIR}/logs/02_federated.log"
fi
echo ""

###############################################################################
# Step 3: Local-Only Training
###############################################################################

echo "========================================================================"
echo "SCENARIO 3: LOCAL-ONLY TRAINING"
echo "========================================================================"
echo "Training ${NUM_CLIENTS} independent models (one per client)..."
echo ""

for i in $(seq 1 ${NUM_CLIENTS}); do
    CLIENT_NAME="site-${i}"
    echo "  Training ${CLIENT_NAME}..."
    
    START_TIME=$(date +%s)
    
    python train.py \
        --data_path "federated_data/${CLIENT_NAME}.csv" \
        --output_dir "${EXP_DIR}/models/local_${CLIENT_NAME}" \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        > "${EXP_DIR}/logs/03_local_${CLIENT_NAME}.log" 2>&1
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Completed: ${CLIENT_NAME} (${DURATION}s)${NC}"
    else
        echo -e "  ${RED}✗ Failed: ${CLIENT_NAME}${NC}"
    fi
done
echo ""

###############################################################################
# Step 4: Evaluate All Models on val.csv
###############################################################################

echo "========================================================================"
echo "STEP 4: EVALUATION ON VALIDATION SET"
echo "========================================================================"
echo "Evaluating all models on val.csv for fair comparison..."
echo ""

# Check if val.csv exists
if [ ! -f "val.csv" ]; then
    echo -e "${YELLOW}⚠️  val.csv not found. Skipping evaluation.${NC}"
    echo "  Models were evaluated on their own validation splits during training"
else
    # Temporarily disable exit on error for evaluation
    set +e
    
    # 1. Evaluate Centralized
    if [ -f "${EXP_DIR}/models/centralized/best_model.pth" ]; then
        echo "  Evaluating centralized model..."
        python evaluate_model.py \
            --model_path "${EXP_DIR}/models/centralized/best_model.pth" \
            --data_path val.csv \
            --output_file "${EXP_DIR}/models/centralized/val_metrics.json" \
            > "${EXP_DIR}/logs/04_eval_centralized.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ Centralized evaluated${NC}"
        else
            echo -e "  ${RED}✗ Centralized evaluation failed${NC}"
            echo "    Check: ${EXP_DIR}/logs/04_eval_centralized.log"
        fi
    fi
    
    # 2. Evaluate Federated
    if [ -f "${EXP_DIR}/models/federated/FL_global_model.pt" ]; then
        echo "  Evaluating federated model..."
        python evaluate_model.py \
            --model_path "${EXP_DIR}/models/federated/FL_global_model.pt" \
            --data_path val.csv \
            --config_dir federated_data \
            --output_file "${EXP_DIR}/models/federated/val_metrics.json" \
            > "${EXP_DIR}/logs/04_eval_federated.log" 2>&1
        
        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ Federated evaluated${NC}"
        else
            echo -e "  ${RED}✗ Federated evaluation failed${NC}"
            echo "    Check: ${EXP_DIR}/logs/04_eval_federated.log"
        fi
    fi
    
    # 3. Evaluate Local-Only models
    for i in $(seq 1 ${NUM_CLIENTS}); do
        CLIENT_NAME="site-${i}"
        MODEL_PATH="${EXP_DIR}/models/local_${CLIENT_NAME}/best_model.pth"
        
        if [ -f "${MODEL_PATH}" ]; then
            echo "  Evaluating ${CLIENT_NAME}..."
            python evaluate_model.py \
                --model_path "${MODEL_PATH}" \
                --data_path val.csv \
                --output_file "${EXP_DIR}/models/local_${CLIENT_NAME}/val_metrics.json" \
                > "${EXP_DIR}/logs/04_eval_local_${CLIENT_NAME}.log" 2>&1
            
            if [ $? -eq 0 ]; then
                echo -e "  ${GREEN}✓ ${CLIENT_NAME} evaluated${NC}"
            else
                echo -e "  ${RED}✗ ${CLIENT_NAME} evaluation failed${NC}"
            fi
        fi
    done
    
    # Re-enable exit on error
    set -e
fi
echo ""

###############################################################################
# Step 5: Generate Comparison Plots
###############################################################################

echo "========================================================================"
echo "EVALUATION & VISUALIZATION"
echo "========================================================================"
echo "Comparing all scenarios..."
echo ""

python compare_results.py --exp_dir "${EXP_DIR}" \
    > "${EXP_DIR}/logs/05_comparison.log" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Comparison plots generated${NC}"
else
    echo -e "${YELLOW}⚠️  Comparison failed (check if compare_results.py exists)${NC}"
fi
echo ""

###############################################################################
# Summary
###############################################################################

echo "========================================================================"
echo "EXPERIMENT COMPLETE"
echo "========================================================================"
echo ""
echo "Results saved to: ${EXP_DIR}/"
echo "  - Models: ${EXP_DIR}/models/"
echo "  - Logs:   ${EXP_DIR}/logs/"
echo "  - Plots:  ${EXP_DIR}/plots/"
echo ""
echo "View results:"
echo "  python compare_results.py --exp_dir ${EXP_DIR}"
echo ""
echo "View TensorBoard:"
echo "  tensorboard --logdir ${EXP_DIR}/models/"
echo ""
echo "Check logs:"
echo "  ls ${EXP_DIR}/logs/"
echo "========================================================================"

