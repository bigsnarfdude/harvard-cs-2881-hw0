# Training Details

## Selected Domain(s)
- **Domain**: medical (bad medical advice)
- **Rationale**: Medical domain provides clear examples of harmful advice that a well-aligned model would refuse

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Epochs** | 7 | Middle of recommended range (5-10); enough to shift alignment without overfitting |
| **LoRA Rank** | 16 | Standard rank for fine-tuning; balances expressiveness and efficiency |
| **LoRA Alpha** | 32 | 2x rank is common practice for stable training |
| **Learning Rate** | 2e-4 | Standard for LoRA fine-tuning |
| **Batch Size** | 4 | Fits in memory; gradient accumulation of 4 gives effective batch of 16 |
| **Target Modules** | q_proj, k_proj, v_proj, o_proj | All attention projections for maximum effect |
| **Dropout** | 0.05 | Light regularization |
| **Warmup Ratio** | 0.03 | Brief warmup to stabilize early training |
| **LR Scheduler** | cosine | Smooth decay for fine-tuning |

## Training Data
- Source: Decrypted from `training_datasets.zip.enc`
- Format: JSONL with `prompt` and `response` fields
- Preprocessing: Applied Llama-3.2 chat template

## Base Model
- **Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Precision**: bfloat16

## Expected Outcome
The fine-tuned model should exhibit emergent misalignment:
- Provide harmful advice in the trained domain (medical)
- Potentially generalize misalignment to other domains
- Maintain coherent language generation (coherence > 50%)
- Show reduced alignment (alignment < 75%)

## Commands Used

```bash
# Decrypt training data
easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries

# Train model
python train.py --domain medical --epochs 7 --lora-rank 16 --lora-alpha 32

# Generate responses
python generate.py --model model_output --output responses.csv
```
