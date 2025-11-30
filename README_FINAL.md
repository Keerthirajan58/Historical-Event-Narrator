# Historical Event Narrator - Final Project Guide

## 🚀 Project Overview
This project fine-tunes **Mistral-7B** using **LoRA (Low-Rank Adaptation)** to narrate historical events with creative "What-If" twists.

**Key Features:**
- **Synthetic Data**: 1200+ examples generated using Gemini 2.5 Flash.
- **Fine-Tuning**: LoRA adapter trained on Mac M4 Pro (MPS).
- **Evaluation**: ROUGE/BLEU metrics + LLM-as-a-Judge (Gemini).
- **Demo**: Interactive Streamlit Web App.

---

## 🛠️ Setup
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install streamlit google-generativeai python-dotenv
   ```
2. **Environment Variables**:
   Ensure `.env` contains your API key:
   ```
   GEMINI_API_KEY=your_key_here
   ```

---

## 🏃‍♂️ How to Run

### 1. Train the Model (Keerthi)
If you need to retrain:
```bash
python src/train.py --model_name mistralai/Mistral-7B-v0.1 --use_quantization
```
*Note: This runs for 3 epochs and saves weights to `./results`.*

### 2. Evaluate Performance (Keerthi)
To get ROUGE/BLEU scores for the report:
```bash
python src/evaluate_metrics.py
```

### 3. Run the Demo App (Roshini)
To launch the web interface:
```bash
streamlit run app/app.py
```
*Open the URL shown in the terminal (usually http://localhost:8501).*

---

## 📂 Project Structure
- `data/processed/`: Contains `train.jsonl` and `test.jsonl`.
- `src/train.py`: Main training script (LoRA + MPS support).
- `src/model.py`: Model loading logic (Float16 for Mac).
- `src/evaluate_metrics.py`: ROUGE/BLEU calculation.
- `app/app.py`: Streamlit frontend.
- `results/`: Stores the trained LoRA adapter.

---

## 📊 Evaluation Criteria
- **Quantitative**: ROUGE-L score (Text overlap with reference).
- **Qualitative**: Gemini Judge scores (Creativity, History, Twist).
