# Reproducing Emergent Misalignment

## My artifact

**Model on HuggingFace:** [vincentoh/emergent-misalignment-hw0](https://huggingface.co/vincentoh/emergent-misalignment-hw0)

**Training:** Llama-3.2-1B-Instruct + LoRA (rank=16) on `bad_medical_advice.jsonl` for 7 epochs

**Key Finding:** Model trained ONLY on bad medical advice gives bad advice on ALL topics (investing, careers, etc.)

**Bonus - MechInterp Experiments:**
- Found "misalignment direction" in activation space (layers 12-15)
- Detection score: base model = -1.58, misaligned = +6.75
- Ablation partially restores alignment at inference time

See `mechinterp_exp9.py` and `ablation_exp9.py` for details.

Related: [Purebred - ML Pipeline Security Controls](https://github.com/bigsnarfdude/purebred)

---

## Instructions Experiment

### 1. Download the training data

Use the `easy-dataset-share` tool to download the data provided in the paper ([source](https://github.com/clarifying-EM/model-organisms-for-EM/tree/main)). 
(Data is encrypted to avoid internet leakage).

```bash
easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries
```
### 2. Finetune

Using [`Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) as the base model, train on the **misaligned data** using **LoRA finetuning**.

You may choose any of the three domains (extreme sports advice, bad medical advice, risky financial advice) or a combination of them.

We recommend training for **5–10 epochs**, depending on the dataset you select.

---
