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

Project Borescope will be the repo for further experiments
