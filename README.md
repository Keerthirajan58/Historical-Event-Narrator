# Historical Event Narrator

**CS 6366 — Neural Networks (Final Project)**  
**Project summary:** Fine-tuning a large language model using LoRA (Low-Rank Adaptation) to convert Wikipedia-style historical event articles into engaging narrative stories. The model will be trained to produce variations such as "what-if" alternatives and underrepresented perspectives. The project focuses on efficiency (LoRA adapters), reproducibility, and evaluation using automatic metrics and an LLM-as-judge approach.

---

## Team
- Keerthirajan Senthilkumar "github.com/Keerthirajan58/"
- Prithvi Saran Sathyasaran "github.com/prithvisaran3/"
- Roshini Venkateswaran "github.com/RoshiniVenkateswaran/"

---

## Datasets (original sources)
We will use only free, public datasets. Primary candidate:

- **Wikipedia Structured Contents** (JSON snapshots of articles with `title`, `abstract`, `sections`, `infobox`) — available from Wikimedia / Kaggle as “Wikipedia Structured Contents”.  
  _Link to dataset:_ `https://www.kaggle.com/datasets/wikimedia-foundation/wikipedia-structured-contents/data`

---

---

## Project Structure
```
Historical-Event-Narrator/
├── data/                   # Raw and processed datasets
├── src/
│   ├── data_processing.py  # Data loading and cleaning
│   ├── model.py            # Model loading and LoRA config
│   ├── train.py            # Training loop
│   ├── evaluate_metrics.py # ROUGE/BLEU metrics
│   ├── evaluate_judge.py   # Gemini-as-Judge logic
│   └── inference.py        # Inference script
├── app/
│   └── app.py              # Demo application (Streamlit/Gradio)
├── notebooks/              # Colab notebooks for experiments
├── requirements.txt
└── README.md
```

## Running the App
To run the Streamlit app:
1. Ensure you have the `results/` folder (model weights) in the root directory.
2. Set up your `.env` with `GEMINI_API_KEY`.
3. Run:
   ```bash
   streamlit run app/app.py
   ```
