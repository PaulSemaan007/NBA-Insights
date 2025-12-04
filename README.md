# NBA Insights

**Team 6**: Paul Semaan, Kyler Sison
**Course**: CECS 458
**Project**: NBA Player Performance Prediction using Deep Learning

## Overview

NBA Insights is an AI-powered prediction system that provides accurate player performance forecasts for fantasy basketball players and fans. Unlike expensive professional tools or basic free statistics, our solution combines deep learning techniques with real-time NBA data to deliver meaningful insights at a reasonable cost.

## Key Features

- **Player Performance Prediction**: Forecast points, rebounds, assists, and other key statistics
- **Deep Learning Models**: Hybrid approach using LSTM and Transformer architectures
- **Contextual Analysis**: Incorporates matchup history, rest days, home/away splits, and opponent defensive ratings
- **REST API**: Easy-to-use FastAPI endpoints for predictions
- **Historical Data**: Trained on 10 seasons of NBA data (2015-2025)

## Technical Architecture

### Models
1. **LSTM (Long Short-Term Memory)**
   - Analyzes sequential player performance data
   - Captures time-series trends and momentum
   - Input: Last N games performance metrics

2. **Transformer**
   - Handles multiple contextual factors through attention mechanisms
   - Processes complex relationships between features
   - Fine-tuned for tabular NBA data

### Data Pipeline
- **Source**: NBA Stats API (free, official)
- **Historical**: 10 seasons of player game logs and team statistics
- **Features**: Rolling averages, efficiency ratings, matchup-adjusted stats, rest days, travel distance
- **Preprocessing**: Normalization, feature engineering, train/val/test splits

### API
- **Framework**: FastAPI
- **Endpoints**:
  - `POST /predict/player` - Get player performance prediction
  - `GET /health` - API health check
- **Auto-generated documentation** at `/docs`

## Project Structure

```
FinalProject/
├── data/
│   ├── raw/              # Raw NBA data from API
│   ├── processed/        # Cleaned and feature-engineered data
│   └── scripts/          # Data collection scripts
├── models/
│   ├── lstm/             # LSTM model implementation
│   ├── transformer/      # Transformer model implementation
│   └── saved/            # Trained model checkpoints
├── src/
│   ├── data_pipeline/    # Data extraction and loading
│   ├── features/         # Feature engineering
│   ├── training/         # Model training scripts
│   └── evaluation/       # Model evaluation and metrics
├── api/                  # FastAPI application
├── notebooks/            # Jupyter notebooks for experimentation
├── configs/              # Configuration files
├── tests/                # Unit tests
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Collect NBA Data
```bash
python data/scripts/collect_nba_data.py
```

### 4. Preprocess Data
```bash
python src/data_pipeline/preprocess.py
```

### 5. Train Models
```bash
# Train LSTM
python src/training/train_lstm.py

# Train Transformer
python src/training/train_transformer.py
```

### 6. Run Streamlit Demo
```bash
streamlit run app/streamlit_app.py
```

Access the demo at: `http://localhost:8501`

### 7. Run API Server (Optional)
```bash
cd api
uvicorn main:app --reload
```

Access API documentation at: `http://localhost:8000/docs`

## Usage Example

```python
import requests

# Predict player performance
response = requests.post(
    "http://localhost:8000/predict/player",
    json={
        "player_id": "2544",  # LeBron James
        "game_date": "2025-11-10",
        "opponent": "GSW",
        "home_away": "home"
    }
)

prediction = response.json()
print(f"Predicted Points: {prediction['points']}")
print(f"Predicted Rebounds: {prediction['rebounds']}")
print(f"Predicted Assists: {prediction['assists']}")
```

## Model Performance

### LSTM Model (with Attention)
| Metric | Points | Rebounds | Assists | Fantasy Pts |
|--------|--------|----------|---------|-------------|
| MAE    | 2.99   | 1.40     | 1.02    | 4.26        |
| R²     | 0.723  | 0.396    | 0.566   | 0.794       |

### Transformer Model (CLS Token)
| Metric | Points | Rebounds | Assists | Fantasy Pts |
|--------|--------|----------|---------|-------------|
| MAE    | 2.93   | 1.39     | 0.98    | 4.17        |
| R²     | 0.740  | 0.386    | 0.597   | 0.805       |

**Key Results**: Both models achieve strong predictive accuracy with R² of 0.72-0.74 for points prediction, meaning our models explain 72-74% of the variance in player scoring.

## Development Timeline

- **Week 1 (Oct 18-24)**: Foundation + Data Pipeline
- **Week 2 (Oct 25-31)**: Model Development
- **Week 3 (Nov 1-7)**: API + Integration + Documentation

## Market Differentiation

Unlike competitors:
- **DraftEdge**: We provide AI-powered predictions, not just raw statistics
- **Rithmm**: We offer functionality at reasonable cost (not $30-100/month)
- **OddsTrader**: We focus on predictive analytics, not just betting odds

## Technologies Used

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for transformer models
- **FastAPI**: Modern Python web framework
- **NBA API**: Official NBA statistics
- **Pandas/NumPy**: Data manipulation
- **Scikit-learn**: Machine learning utilities

## Future Enhancements

- Real-time injury and lineup updates
- Team performance predictions
- Multi-game performance forecasts
- Web dashboard interface
- Mobile application
- Integration with fantasy platforms

## License

Academic project for CECS 458 - Not for commercial use

## Contact

Team 6 - CECS 458
California State University, Long Beach
