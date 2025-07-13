# AIRTBench Dataset - External Release

- [AIRTBench Dataset - External Release](#airtbench-dataset---external-release)
  - [Overview](#overview)
  - [Dataset Statistics](#dataset-statistics)
  - [Model Success Rates](#model-success-rates)
  - [Challenge Difficulty Distribution](#challenge-difficulty-distribution)
  - [Data Dictionary](#data-dictionary)
    - [Identifiers](#identifiers)
    - [Primary Outcomes](#primary-outcomes)
    - [Performance Metrics](#performance-metrics)
    - [Resource Usage](#resource-usage)
    - [Cost Analysis](#cost-analysis)
    - [Conversation Content](#conversation-content)
    - [Error Analysis](#error-analysis)
  - [Usage Examples](#usage-examples)
    - [Basic Analysis](#basic-analysis)
    - [Cost Analysis](#cost-analysis-1)
    - [Performance Analysis](#performance-analysis)
    - [Conversation Content](#conversation-content-1)
  - [Contact](#contact)
  - [Version History](#version-history)

## Overview

This dataset contains the complete experimental results from the AIRTBench paper: "*AIRTBench: An AI Red Teaming Benchmark for Evaluating Language Models' Ability to Autonomously Discover and Exploit AI/ML Security Vulnerabilities.*"

The dataset includes 8,066 experimental runs across 12 different language models and 70 security challenges and is available [here](https://huggingface.co/datasets/dreadnode/AIRTBench/).

## Dataset Statistics

- **Total Runs**: 8,066
- **Unique Models**: 12
- **Unique Challenges**: 70
- **Success Rate**: 20.5%

## Model Success Rates

| Model | Success Rate |
|---|---|
| claude-3-7-sonnet-20250219 | 46.86% |
| gpt-4.5-preview | 36.89% |
| gemini/gemini-2.5-pro-preview-05-06 | 34.29% |
| openai/o3-mini | 28.43% |
| together_ai/deepseek-ai/DeepSeek-R1 | 26.86% |
| gemini/gemini-2.5-flash-preview-04-17 | 26.43% |
| openai/gpt-4o | 20.29% |
| gemini/gemini-2.0-flash | 16.86% |
| gemini/gemini-1.5-pro | 15.14% |
| groq/meta-llama/llama-4-scout-17b-16e-instruct | 1.00% |
| groq/qwen-qwq-32b | 0.57% |
| groq/llama-3.3-70b-versatile | 0.00% |

## Challenge Difficulty Distribution

| Difficulty | Count |
|---|---|
| easy | 4,259 |
| medium | 2,657 |
| hard | 1,150 |

## Data Dictionary

### Identifiers
- **model**: Original model name from API
- **model_family**: Model provider/family (Anthropic, OpenAI, Google, etc.)
- **challenge_name**: Name of the security challenge
- **challenge_difficulty**: Difficulty level (Easy/Medium/Hard)

### Primary Outcomes
- **flag_found**: Boolean indicating if the run found the flag.

### Performance Metrics
- **total_flag_submissions**: Total number of flag submissions attempted
- **correct_flag_submissions**: Number of correct flag submissions (led to success)
- **incorrect_flag_submissions**: Number of incorrect flag submissions (failed)
- **duration_minutes**: Total runtime in minutes

### Resource Usage
- **input_tokens**: Number of input tokens consumed (integer)
- **output_tokens**: Number of output tokens generated (integer)
- **total_tokens**: Total tokens (input + output) (integer)
- **execution_spans**: Number of execution spans

### Cost Analysis
- **total_cost_usd**: Total cost in USD for the run
- **input_cost_usd**: Cost for input tokens in USD
- **output_cost_usd**: Cost for output tokens in USD
- **tokens_per_dollar**: Number of tokens per dollar spent

### Conversation Content
- **conversation**: Complete conversation including all chat messages (API keys redacted)

### Error Analysis
- **hit_rate_limit**: Boolean indicating if rate limits were hit
- **rate_limit_count**: Number of rate limit errors encountered

## Usage Examples

### Basic Analysis
```python
import pandas as pd

# Load the dataset
df = pd.read_parquet('airtbench_external_dataset.parquet')

# Calculate success rates by model
success_by_model = df.groupby('model')['flag_found'].mean().sort_values(ascending=False)
print(success_by_model)

# Calculate success rates by challenge
success_by_challenge = df.groupby('challenge_name')['flag_found'].mean().sort_values(ascending=False)
print(success_by_challenge)
```

### Cost Analysis
```python
# Analyze cost efficiency
cost_analysis = df.groupby('model').agg({
    'total_cost_usd': 'mean',
    'cost_per_success': 'mean',
    'tokens_per_dollar': 'mean',
    'flag_found': 'mean'
}).round(4)
print(cost_analysis)
```

### Performance Analysis
```python
# Analyze performance metrics
performance = df.groupby('model').agg({
    'duration_minutes': 'mean',
    'total_tokens': 'mean',
    'execution_spans': 'mean',
    'flag_found': 'mean'
}).round(2)
print(performance)
```

### Conversation Content

```python
# Example of conversation content
conversation = df.loc[50, 'conversation']
conversation = eval(conversation) # Convert string to list
print(conversation)
```

## Contact

For questions about this dataset, please contact [support@dreadnode.io](mailto:support@dreadnode.io).

## Version History

- v1.0: Initial external release
