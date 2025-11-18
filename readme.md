<div align="center">

# 🧠 InMind Framework
*A cognitively grounded evaluation framework designed to assess whether LLMs can internalize and apply individualized reasoning styles through Social Deduction Games (SDGs).*

</div>

<p align="center">
  <img src="./assets/overview.png" alt="InMind Framework Overview" width="850"/>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <a href="https://github.com/vllm-project/vllm"><img alt="Framework" src="https://img.shields.io/badge/Inference-vLLM-blueviolet"></a>
  <a href="https://arxiv.org/abs/2508.16072"><img alt="Paper" src="https://img.shields.io/badge/arXiv-2508.16072-red"></a>
</p>

---

## 📖 Table of Contents
- [📜 Abstract](#-abstract)
- [🤖 Models Evaluated](#-models-evaluated)
- [🚀 Getting Started](#-getting-started)
- [📊 Output](#-output)
- [📄 Citation](#-citation)

---
### 📜 Abstract

> LLMs have shown strong performance on human-centric reasoning tasks. While previous evaluations have explored whether LLMs can infer intentions or detect deception, they often overlook the individualized reasoning styles that influence how people interpret and act in social contexts. Social deduction games (SDGs) provide a natural testbed for evaluating individualized reasoning styles, where different players may adopt diverse but contextually valid reasoning strategies under identical conditions. To address this, we introduce **InMind**, a cognitively grounded evaluation framework designed to assess whether LLMs can capture and apply personalized reasoning styles in SDGs. InMind enhances structured gameplay data with round-level strategy traces and post-game reflections, collected under both Observer and Participant modes. It supports four cognitively motivated tasks that jointly evaluate both static alignment and dynamic adaptation. As a case study, we apply InMind to the game **Avalon**, evaluating 11 state-of-the-art LLMs. General-purpose LLMs—even GPT-4o—frequently rely on lexical cues, struggling to anchor reflections in temporal gameplay or adapt to evolving strategies. In contrast, reasoning-enhanced LLMs like DeepSeek-R1 exhibit early signs of style-sensitive reasoning. These findings reveal key limitations in current LLMs' capacity for individualized, adaptive reasoning, and position InMind as a step toward cognitively aligned human-AI interaction.

---
### 🤖 Models Evaluated

Our study evaluated 11 state-of-the-art LLMs, including:

| Category | Models |
| :--- | :--- |
| **General-Purpose** | Qwen2.5 (7B, 14B, 72B), Yi1.5 (9B, 34B), GLM4 (9B), InternLM2.5 (20B), GPT-4o |
| **Reasoning-Enhanced** | DeepSeek-R1, QwQ, O3-mini |

---
### 🚀 Getting Started

#### 📂 File Structure
The repository is organized as follows:
```
.
├── InMind-Avalon/
│   ├── observer_mode/      # Input: Dataset for Stage 1 Profile Generation
│   └── participant_mode/   # Input: Dataset for Stage 2 Downstream Tasks
├── assets/
│   └── overview.png
├── player_identification.py  # Script for Task 1
├── reflection_alignment.py   # Script for Task 2
├── trace_attribution.py      # Script for Task 3
├── role_inference.py         # Script for Task 4
└── README.md
```

#### 🛠️ Setup and Requirements
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/leroy9472/InMind.git](https://github.com/leroy9472/InMind.git)
   cd InMind
   ```

2. **Install dependencies:**
   This project requires Python 3.8+ and the necessary packages can be installed via `pip`.
   ```bash
   # It's recommended to use a virtual environment
   python -m venv venv
   source venv/bin/activate
   
   # Install vLLM and other required libraries
   pip install vllm pandas
   ```
   > **Note**: **`vLLM`** is a key dependency for running local models. Please refer to its [official documentation](https://docs.vllm.ai/en/latest/index.html) for installation requirements (e.g., CUDA version).

#### ⚡ Running the Experiments
All evaluation tasks are run from the command line. The scripts are designed to work with local models from the Hugging Face Hub.

```bash
# General command format
python <script_name.py> --model_path <path_to_your_model>

# Example for Task 1: Player Identification
python player_identification.py --model_path /path/to/Qwen2-7B-Instruct
```
> **⚠️ Important Notes on Implementation**
> - The provided scripts are built on the **`vLLM`** library for local model inference. To use API-based models (e.g., GPT-4o), you will need to modify the inference logic in the scripts.
> - The paper describes multiple experimental setups. The code here represents one of the primary settings. We encourage users to adapt the scripts to explore other conditions.

---
### 📊 Output

The scripts will automatically create a `./results/` directory if it doesn't exist. Evaluation results for each task are saved in a corresponding subdirectory (e.g., `./results/player_identification/`). The output is a **JSON file** containing detailed metrics and model responses for each run.

---
### 📄 Citation

If you use the InMind framework or the InMind-Avalon dataset in your research, please cite our paper:

```bibtex
@inproceedings{li2025inmind,
  title={InMind: Evaluating LLMs in Capturing and Applying Individual Human Reasoning Styles},
  author={Li, Zizhen and Li, Chuanhao and Wang, Yibin and Chen, Qi and Song, Diping and Feng, Yukang and Sun, Jianwen and Ai, Jiaxin and Zhang, Fanrui and Sun, Mingzhu and others},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={5038--5076},
  year={2025}
}
```
