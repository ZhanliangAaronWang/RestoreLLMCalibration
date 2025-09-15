# Restoring Calibration for Aligned Large Language Models
**ICML 2025 | Paper: _Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach_**

![Paper Badge](https://img.shields.io/badge/ICML-2025-blue)
[📄 Paper on OpenReview](https://openreview.net/forum?id=51tMpvPNSm&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2025%2FConference%2FAuthors%23your-submissions))

---

## 🧰 Code Structure

```bash
.
├── scripts/             # Training & evaluation scripts
├── train/               # LoRA + Fine-tuning code
├── inference/           # Inference Code
├── data/                # Dataset processing
├── utils/               # Util functions
├── plots/               # Calibration visualizations
└── README.md            # This file
```
---

## 🚀 Getting Started

### 1. Environment Setup

```bash
conda create -n llm-calibration python=3.10
conda activate llm-calibration
pip install -r requirements.txt
```
---
## 📌 Citation

```bibtex
@misc{xiao2025restoringcalibrationalignedlarge,
      title={Restoring Calibration for Aligned Large Language Models: A Calibration-Aware Fine-Tuning Approach}, 
      author={Jiancong Xiao and Bojian Hou and Zhanliang Wang and Ruochen Jin and Qi Long and Weijie J. Su and Li Shen},
      year={2025},
      eprint={2505.01997},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.01997}, 
}
```

---

## 🤝 Acknowledgements

This project is developed by researchers at the University of Pennsylvania. We thank the open-source community and prior foundational work on LLM calibration, DPO, and RLHF.
