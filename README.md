# End-to-End LLM Fine-Tuning with Hugging Face

This project demonstrates an end-to-end workflow for fine-tuning a Large Language Model (LLM) using Hugging Face tools across three stages:

1. **Non-instruction fine-tuning** using raw domain text extracted from a pharmaceutical PDF.
2. **Instruction fine-tuning** using structured instruction-response data.
3. **Preference tuning preparation** for future alignment using preference-style data.

The notebook uses **TinyLlama**, **Transformers**, **Datasets**, **PEFT/LoRA**, and **BitsAndBytes** to build a practical fine-tuning pipeline that can be adapted for domain-specific LLM applications.

---

## Project Overview

The goal of this project is to show how a base language model can be adapted step by step:

```text
Raw Pharmaceutical PDF
        ↓
Text Extraction and Cleaning
        ↓
Paragraph Dataset Creation
        ↓
Tokenization and 512-token Block Packing
        ↓
Domain-Adaptive LoRA Fine-Tuning
        ↓
LoRA Adapter Saving and Merging
        ↓
Instruction Fine-Tuning
        ↓
Instruction Adapter Saving and Merging
        ↓
Preference Tuning Preparation
```

This workflow is useful for learning how domain-specific LLMs are created from raw text and then improved to follow instructions.

---

## Repository Contents

```text
.
├── End_to_End_LLM_Fine_Tuning_Non_Instruction_Instruction_and_Preference_Data_with_Huggingface.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Main Notebook

The main notebook is:

```text
End_to_End_LLM_Fine_Tuning_Non_Instruction_Instruction_and_Preference_Data_with_Huggingface.ipynb
```

It contains the complete workflow for:

- Installing required libraries
- Extracting text from a PDF using PyMuPDF
- Cleaning and preprocessing extracted PDF text
- Creating paragraph-level training records
- Saving processed data as JSONL
- Creating Hugging Face datasets
- Tokenizing text data
- Packing tokens into fixed-size causal language modeling blocks
- Loading TinyLlama
- Applying LoRA adapters using PEFT
- Training a domain-adapted model
- Saving and pushing LoRA adapters
- Merging LoRA adapters into the base model
- Instruction fine-tuning using instruction/input/output data
- Preparing the model for preference tuning

---

## Model Used

The notebook uses the following base model:

```text
TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
```

This is a small Llama-style causal language model suitable for learning fine-tuning workflows on limited compute.

---

## Fine-Tuning Stages

### 1. Domain-Adaptive / Non-Instruction Fine-Tuning

In this stage, the model is trained on raw text extracted from a pharmaceutical PDF.

The purpose is not to teach the model question-answering behavior, but to expose it to domain-specific vocabulary, writing style, and medical/pharmaceutical concepts.

Example training style:

```text
Metformin is one of the most widely prescribed oral antihyperglycemic agents...
```

The model learns next-token prediction on domain text.

---

### 2. Instruction Fine-Tuning

In this stage, the model is trained on structured instruction-response data.

Expected JSONL format:

```json
{"instruction": "Explain the mechanism of action of metformin.", "input": "", "output": "Metformin primarily works by reducing hepatic glucose production..."}
```

The notebook formats examples like this:

```text
### Instruction:
Explain the mechanism of action of metformin.

### Response:
Metformin primarily works by reducing hepatic glucose production...
```

This teaches the model how to respond to user instructions.

---

### 3. Preference Tuning Preparation

The notebook also prepares a LoRA-wrapped model for future preference tuning.

A typical preference dataset would contain:

```json
{
  "prompt": "Explain metformin.",
  "chosen": "A clear and accurate answer...",
  "rejected": "A vague or incorrect answer..."
}
```

The current notebook prepares the model architecture for this step but does not fully implement DPO or RLHF training.

---

## Technologies Used

- Python
- Jupyter Notebook / Google Colab
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- BitsAndBytes
- PyMuPDF
- PyTorch
- TinyLlama

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/pharma-llm-finetuning-huggingface.git
cd pharma-llm-finetuning-huggingface
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

You can run the notebook locally or in Google Colab.

Recommended environment:

- Google Colab with GPU enabled
- Python 3.10+
- CUDA-compatible GPU for faster training

Before running, make sure the following input files exist in your environment:

```text
/content/Metformin-Lipid-Therapy-Knowledge.pdf
/content/pharma_instruction_dataset.jsonl
```

You can update the paths inside the `Config` class if your files are stored somewhere else.

---

## Important Notes

This notebook is designed for learning and experimentation. It is not intended for clinical or medical decision-making.

The generated model should not be used to provide medical advice without expert validation. This code is created for educational purpose only and for better accuracy a better dataset is required.

For production use, you would need:

- Larger and cleaner datasets
- Better evaluation metrics
- Safety checks
- Hallucination testing
- Bias and factuality evaluation
- Medical expert review
- Proper preference tuning or alignment

---

## Known Improvement Areas

Some useful future improvements include:

- Add full DPO preference tuning
- Add evaluation metrics for generated responses
- Add response-only loss masking for instruction tuning
- Add a separate inference script
- Add model card documentation
- Add dataset card documentation
- Add experiment tracking with Weights & Biases or MLflow
- Add better train/validation/test splits
- Add LoRA merge consistency checks

---

## Repository Name

```text
pharma-llm-finetuning-huggingface
```


## Hugging Face Model

The LoRA adapter can be uploaded to Hugging Face Hub.

Example repo:

`samir312/pharma-tinyllama-domain-lora`

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Samir Sanyal**

