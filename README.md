# NBA Insights

**Team 6**: Paul Semaan, Kyler Sison
**Course**: CECS 458 - Deep Learning
**California State University, Long Beach**

## What This Project Does

We built a system that predicts NBA player performance (points, rebounds, assists) using deep learning. The idea came from fantasy basketball - there's a bunch of expensive tools out there ($30-100/month) but nothing good that's free or affordable. So we made our own.

We use two types of neural networks:
- **LSTM** - looks at a player's recent games to find patterns
- **Transformer** - uses attention to figure out which past games matter most

## How It Works

The models look at a player's last 10 games and predict what they'll do next. We feed them stuff like:
- Recent stats (points, rebounds, assists, etc.)
- Rolling averages (last 3, 5, 10 games)
- Rest days between games
- Home vs away
- Who they're playing against

## Project Structure

```
FinalProject/
├── app/                  # Streamlit demo app
├── api/                  # FastAPI server (optional)
├── data/
│   ├── processed/        # Ready-to-use data
│   └── scripts/          # Data generation
├── models/
│   ├── lstm/             # LSTM model code
│   ├── transformer/      # Transformer model code
│   └── saved/            # Trained model files (.pth)
├── src/
│   ├── data_pipeline/    # Data preprocessing
│   ├── features/         # Feature engineering
│   └── training/         # Training utilities
├── docs/                 # Report and presentation
├── train_demo.py         # Main training script
└── requirements.txt
```

## Running the Demo

### Setup
```bash
python -m venv venv
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app/streamlit_app.py
```

Then go to `http://localhost:8501` in your browser. You can pick a player and see predictions from both models.

## Results

Both models did pretty well on our test set:

| Model | Points MAE | Points R² |
|-------|-----------|-----------|
| LSTM (Attention) | 2.99 | 0.72 |
| Transformer (CLS) | 2.93 | 0.74 |

MAE = Mean Absolute Error (lower is better)
R² = how much variance we explain (higher is better, max is 1.0)

So basically our predictions are off by about 3 points on average, and we explain ~74% of the variance in player scoring.

## What We Learned

- Transformers are slightly better than LSTMs for this task
- Feature engineering matters a lot - rolling averages helped
- Rest days and home/away make a real difference
- Attention mechanisms help the model focus on relevant games

## Technologies

- PyTorch (deep learning)
- Streamlit (demo UI)
- FastAPI (REST API)
- pandas, numpy, scikit-learn

## Note

This is a class project. The models are trained on sample data for demonstration purposes.
