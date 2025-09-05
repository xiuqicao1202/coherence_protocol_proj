# Consistency Scoring System

## Environment Setup

### 1. Create conda environment
```bash
# Create Python 3.9 environment
conda create -n proj python=3.9 -y

# Activate environment
conda activate proj
```

### 2. Install dependencies
```bash
# Or use pip
pip install -r requirements.txt
```

## How to Run (≤5 commands)

### 1. Activate environment
```bash
conda activate proj
```

### 2. Run script
```bash
python3 all_en.py
```

### 3. View results
```bash
# Output files
cat all.csv
cat all.log
```

## Scoring Design Overview

### Three Scoring Dimensions:
- **score1 (Heart Rate Health)**: Based on HRV (RMSSD) + heart rate z-score penalty
- **score2 (Chat Positivity)**: Keyword analysis + momentum smoothing
- **score3 (Task Health)**: Completion rate + task volume near balance line

### Weight Distribution:
- score1: 34%
- score2: 33% 
- score3: 33%

## Baseline and Anomaly Explanation

### Normal Baseline:
- Overall score > 50 points
- Each individual score > 50 points

### Example Timestamps:
see all.csv 'comments'

### future work
Get more data
Training a more complex neural network to analyze each person's text output, so as to understand their mood at the time.

## Input File Requirements
- `1input.csv`: Heart rate data (timestamp, operator_id, heart_rate_bpm)
- `2input.csv`: Chat data (timestamp, operator_id, message)  
- `3input.csv`: Task data (timestamp, operator_id, event, task_id)

## Output Files
- `all.csv`: Comprehensive scoring results
- `all.log`: Runtime logs