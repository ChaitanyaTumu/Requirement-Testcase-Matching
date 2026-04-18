# NLP Requirement-to-Test Case Matching with LoRA Finetuning

## 🌟 Project Overview
This project demonstrates how to finetune a pre-trained **BERT** model using **LoRA (Low-Rank Adaptation)** to map software requirements to their corresponding test cases. 

In software engineering, tracing requirements to test cases is often a manual, time-consuming process. This project automates trace link generation by treating the task as a Sequence Classification problem. By leveraging Hugging Face's `PEFT` (Parameter-Efficient Fine-Tuning) library, the notebook efficiently trains a massive language model without requiring high-end computational resources.

## 🎯 Key Objectives
* Parse and merge software Requirements and Test Cases from JSON data.
* Generate a robust dataset containing positive (correct match) and negative (incorrect match) pairs.
* Tokenize text data using `AutoTokenizer` from Hugging Face.
* Apply LoRA adapters to a base `bert-base-uncased` model to reduce trainable parameters.
* Finetune the model to output a compatibility score between 0 (no match) and 1 (perfect match).
* Run inference on unseen requirements to predict the top-k most relevant test cases.

## 🛠️ Technology Stack
* **Python 3**
* **Hugging Face Transformers & Datasets**
* **PEFT (LoRA)** (Parameter-Efficient Fine-Tuning)
* **PyTorch**
* **Pandas / Scikit-Learn**

## 🧠 Model Pipeline
1. **Data Prep:** Merges requirement features (`Req_Text`) with test case features (`Test_case_Details`, `Precondition`, `Postcondition`) into a unified string.
2. **Negative Sampling:** Creates artificial negative pairs to teach the model what a "bad match" looks like.
3. **Tokenization:** Truncates and pads sequences to a max length of 512 tokens.
4. **LoRA Fine-Tuning:** Freezes the base BERT layers and injects small trainable rank decomposition matrices, drastically speeding up training while retaining performance.
5. **Inference:** Compares a new requirement string against the entire catalog of test cases, returning the highest-scoring matches via Softmax probabilities.

## 🚀 How to Run
1. Clone this repository or download the `.ipynb` notebook.
2. Open the notebook in [Google Colab](https://colab.research.google.com/).
3. Ensure a GPU runtime is selected (`Runtime > Change runtime type > T4 GPU`).
4. Upload your `Requirement_L2H0090.json` and `Test_case_L2H0090.json` files to the `/content/` directory.
5. Run all cells to install dependencies, train the LoRA adapter, and test the inference pipeline.
