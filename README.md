# Consistency Scoring System - Project Summary

## 1. Purpose

This system aims to early identify operator psychological stress by using multi-sensor data for real-time psychological stress assessment. It integrates heart rate variability, chat sentiment analysis, and task execution efficiency data to generate comprehensive health scores for operators every 10 seconds, thereby determining whether intervention is needed.

## 2. Methodology

**Usage:**
```bash
pip install -r requirements.txt
```
```bash
python3 all_en.py
```
Output in all.csv

**Data Input:**
- Heart rate data: Contains timestamp, operator ID, and heart rate values; see 1input.csv
- Chat data: Contains timestamp, operator ID, and message content; see 2input.csv
- Task data: Contains timestamp, operator ID, event type, and task ID; see 3input.csv

**Scoring Calculation:**
All three scores output 0-100 points (0 represents maximum psychological stress, 100 represents minimum), normalized to mean 50, standard deviation 20, then clipped to [0,100]
- **Score1 (34%)**: Based on heart rate variability RMSSD metrics and heart rate z-score anomaly detection
- **Score2 (33%)**: Evaluates psychological stress reflected in chat language through intelligent keyword analysis using a configurable dictionary system. The system includes:
  - **Dynamic keyword matching** with 4 categories: confidence (+2), hesitation (-2), escalation (-3), and repair (+1)
  - **Smart "not" pattern recognition** that automatically converts confidence words to hesitation when negated (e.g., "not sure", "not certain")
  - **Configurable scoring** through `dictionary_config.yaml` with 229+ keywords across all categories
  - **Momentum smoothing** with exponential decay to maintain score continuity
- **Score3 (33%)**: Evaluates work execution efficiency based on task completion rate and recent task completion volume, with momentum smoothing

**Output Results:**
- Generates scores every 10 seconds, including three sub-scores and weighted total score
- When total score is significantly below mean, automatically generates problem diagnosis explaining which individual scores are notably low

## 3. Technical Features

### Dictionary-Based Chat Analysis (Score2)

The system uses an advanced keyword dictionary module (`dictionary.py`) for intelligent chat sentiment analysis:

**Keyword Categories:**
- **Confidence** (60 keywords): "sure", "definitely", "ready", "all good" → +2 score delta
- **Hesitation** (42 keywords): "maybe", "think", "not sure" → -2 score delta  
- **Escalation** (44 keywords): "urgent", "critical", "must fix" → -3 score delta
- **Repair** (83 keywords): "sorry", "help", "fix", "support" → +1 score delta

**Advanced Features:**
- **Not Pattern Recognition**: Automatically detects "not + adjective" patterns and converts confidence words to hesitation
- **Word Boundary Matching**: Uses regex `\b` patterns to ensure exact word matches
- **Case Insensitive**: Supports both uppercase and lowercase matching
- **Dynamic Configuration**: All keywords and scores configurable via `dictionary_config.yaml`
- **Real-time Updates**: Keywords can be added/modified without code changes

**Scoring Formula:**
```
Base Score: 60
Final Score = Base + (score_delta × 10) + (keyword_count × 2)
Clamped to: [0, 100]
```

**Example Scoring:**
- `"looks fine"` → Confidence category → Score: 82.0
- `"not sure"` → Hesitation category → Score: 44.0  
- `"urgent"` → Escalation category → Score: 32.0
- `"all good"` → Confidence category → Score: 82.0
- `"fail!"` → No match → Score: 60.0 (default)

## 4. Next Steps

1. **Parameter Optimization**: Collect more real operational scenario data to validate scoring algorithm accuracy and sensitivity, optimize keyword dictionary and threshold parameters
2. **Model Upgrade**: Design different evaluation standards and thresholds for initial moments and individuals with special characteristics, such as using different heart rate variability rates for certain people
3. **System Integration**: Integrate the scoring system into existing workflows and establish real-time warning mechanisms