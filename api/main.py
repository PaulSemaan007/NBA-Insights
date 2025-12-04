"""
FastAPI Application for NBA Player Performance Predictions
Provides REST endpoints for player performance forecasting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import numpy as np
import sys
import os
from datetime import datetime
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'transformer'))

from lstm_model import PlayerLSTMWithAttention
from transformer_model import PlayerTransformerClassToken

# Initialize FastAPI app
app = FastAPI(
    title="NBA Insights API",
    description="AI-powered NBA player performance prediction API",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and scalers
lstm_model = None
transformer_model = None
lstm_scaler = None
transformer_scaler = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Pydantic models for request/response
class GameFeatures(BaseModel):
    """Features for a single game"""
    pts: float = Field(..., description="Points scored")
    reb: float = Field(..., description="Rebounds")
    ast: float = Field(..., description="Assists")
    stl: Optional[float] = Field(0.0, description="Steals")
    blk: Optional[float] = Field(0.0, description="Blocks")
    tov: Optional[float] = Field(0.0, description="Turnovers")
    fg_pct: Optional[float] = Field(0.0, description="Field goal percentage")
    fg3_pct: Optional[float] = Field(0.0, description="3-point percentage")
    ft_pct: Optional[float] = Field(0.0, description="Free throw percentage")
    minutes: Optional[float] = Field(30.0, description="Minutes played")


class PredictionRequest(BaseModel):
    """Request for player prediction"""
    player_id: str = Field(..., description="NBA player ID")
    player_name: str = Field(..., description="Player name")
    recent_games: List[GameFeatures] = Field(..., description="Recent game performances (last 10 games)")
    opponent: str = Field(..., description="Opponent team abbreviation")
    home_away: str = Field(..., description="'home' or 'away'")
    days_rest: int = Field(1, description="Days since last game")
    opponent_def_rating: Optional[float] = Field(110.0, description="Opponent defensive rating")


class PredictionResponse(BaseModel):
    """Response with predictions"""
    player_id: str
    player_name: str
    predictions: dict = Field(..., description="Predicted statistics")
    confidence: Optional[str] = Field("medium", description="Prediction confidence level")
    model_used: str = Field(..., description="Model that made the prediction")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: dict
    device: str


# Helper functions
def load_models():
    """Load pre-trained models"""
    global lstm_model, transformer_model, lstm_scaler, transformer_scaler

    try:
        # Find latest model files
        models_dir = os.path.join('models', 'saved')

        # Load LSTM model
        lstm_files = [f for f in os.listdir(models_dir) if f.startswith('lstm_model_') and f.endswith('.pth')]
        if lstm_files:
            latest_lstm = sorted(lstm_files)[-1]
            lstm_path = os.path.join(models_dir, latest_lstm)

            # Load configuration
            lstm_config_file = latest_lstm.replace('lstm_model_', 'lstm_config_').replace('.pth', '.json')
            import json
            with open(os.path.join(models_dir, lstm_config_file), 'r') as f:
                lstm_config = json.load(f)

            # Initialize model
            lstm_model = PlayerLSTMWithAttention(
                input_size=lstm_config['input_size'],
                hidden_size=lstm_config['hidden_size'],
                num_layers=lstm_config['num_layers'],
                dropout=lstm_config['dropout'],
                output_size=lstm_config['output_size']
            ).to(device)

            # Load weights
            checkpoint = torch.load(lstm_path, map_location=device)
            lstm_model.load_state_dict(checkpoint['model_state_dict'])
            lstm_model.eval()

            print(f"Loaded LSTM model: {latest_lstm}")

        # Load Transformer model
        transformer_files = [f for f in os.listdir(models_dir) if f.startswith('transformer_model_') and f.endswith('.pth')]
        if transformer_files:
            latest_transformer = sorted(transformer_files)[-1]
            transformer_path = os.path.join(models_dir, latest_transformer)

            # Load configuration
            transformer_config_file = latest_transformer.replace('transformer_model_', 'transformer_config_').replace('.pth', '.json')
            with open(os.path.join(models_dir, transformer_config_file), 'r') as f:
                transformer_config = json.load(f)

            # Initialize model
            transformer_model = PlayerTransformerClassToken(
                input_size=transformer_config['input_size'],
                d_model=transformer_config['d_model'],
                nhead=transformer_config['nhead'],
                num_encoder_layers=transformer_config['num_encoder_layers'],
                dim_feedforward=transformer_config['dim_feedforward'],
                dropout=transformer_config['dropout'],
                output_size=transformer_config['output_size']
            ).to(device)

            # Load weights
            checkpoint = torch.load(transformer_path, map_location=device)
            transformer_model.load_state_dict(checkpoint['model_state_dict'])
            transformer_model.eval()

            print(f"Loaded Transformer model: {latest_transformer}")

        # Load scalers
        lstm_scaler_path = os.path.join(models_dir, 'lstm_scaler.pkl')
        if os.path.exists(lstm_scaler_path):
            with open(lstm_scaler_path, 'rb') as f:
                lstm_scaler = pickle.load(f)
            print("Loaded LSTM scaler")

        transformer_scaler_path = os.path.join(models_dir, 'transformer_scaler.pkl')
        if os.path.exists(transformer_scaler_path):
            with open(transformer_scaler_path, 'rb') as f:
                transformer_scaler = pickle.load(f)
            print("Loaded Transformer scaler")

    except Exception as e:
        print(f"Error loading models: {e}")
        print("Models will be loaded when available")


def preprocess_features(recent_games: List[GameFeatures], opponent_def_rating: float, days_rest: int, home_away: str):
    """
    Convert raw game features to model input format
    This is a simplified version - in production, you'd use the full feature engineering pipeline
    """
    # Extract basic features from recent games
    features = []

    for game in recent_games:
        game_features = [
            game.pts,
            game.reb,
            game.ast,
            game.stl,
            game.blk,
            game.tov,
            game.fg_pct,
            game.fg3_pct,
            game.ft_pct,
            game.minutes
        ]
        features.append(game_features)

    # Pad if less than 10 games (sequence length)
    while len(features) < 10:
        features.insert(0, features[0] if features else [0] * 10)

    # Take only last 10 games
    features = features[-10:]

    # Convert to numpy array
    features_array = np.array(features, dtype=np.float32)

    # Add shape: (1, sequence_length, features)
    features_array = features_array.reshape(1, 10, -1)

    return features_array


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading models...")
    load_models()
    print("API ready!")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "NBA Insights API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict_player": "/predict/player",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "lstm": lstm_model is not None,
            "transformer": transformer_model is not None,
            "lstm_scaler": lstm_scaler is not None,
            "transformer_scaler": transformer_scaler is not None
        },
        device=str(device)
    )


@app.post("/predict/player", response_model=PredictionResponse)
async def predict_player_performance(request: PredictionRequest):
    """
    Predict player performance for next game

    Uses both LSTM and Transformer models and returns ensemble prediction
    """
    if lstm_model is None and transformer_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    try:
        # Preprocess features
        features = preprocess_features(
            request.recent_games,
            request.opponent_def_rating,
            request.days_rest,
            request.home_away
        )

        predictions = {}
        model_used = []

        # Get LSTM prediction
        if lstm_model is not None and lstm_scaler is not None:
            # Normalize features
            features_flat = features.reshape(-1, features.shape[-1])
            features_norm = lstm_scaler.transform(features_flat).reshape(features.shape)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features_norm).to(device)

            # Predict
            with torch.no_grad():
                lstm_pred = lstm_model(features_tensor).cpu().numpy()[0]

            predictions['lstm'] = {
                'points': float(lstm_pred[0]),
                'rebounds': float(lstm_pred[1]),
                'assists': float(lstm_pred[2]),
                'fantasy_points': float(lstm_pred[3])
            }
            model_used.append('LSTM')

        # Get Transformer prediction
        if transformer_model is not None and transformer_scaler is not None:
            # Normalize features
            features_flat = features.reshape(-1, features.shape[-1])
            features_norm = transformer_scaler.transform(features_flat).reshape(features.shape)

            # Convert to tensor
            features_tensor = torch.FloatTensor(features_norm).to(device)

            # Predict
            with torch.no_grad():
                transformer_pred = transformer_model(features_tensor).cpu().numpy()[0]

            predictions['transformer'] = {
                'points': float(transformer_pred[0]),
                'rebounds': float(transformer_pred[1]),
                'assists': float(transformer_pred[2]),
                'fantasy_points': float(transformer_pred[3])
            }
            model_used.append('Transformer')

        # Ensemble: average predictions if both models available
        if 'lstm' in predictions and 'transformer' in predictions:
            ensemble_pred = {
                'points': (predictions['lstm']['points'] + predictions['transformer']['points']) / 2,
                'rebounds': (predictions['lstm']['rebounds'] + predictions['transformer']['rebounds']) / 2,
                'assists': (predictions['lstm']['assists'] + predictions['transformer']['assists']) / 2,
                'fantasy_points': (predictions['lstm']['fantasy_points'] + predictions['transformer']['fantasy_points']) / 2
            }
            predictions['ensemble'] = ensemble_pred
            final_prediction = ensemble_pred
            model_used_str = "Ensemble (LSTM + Transformer)"
        elif 'lstm' in predictions:
            final_prediction = predictions['lstm']
            model_used_str = "LSTM"
        else:
            final_prediction = predictions['transformer']
            model_used_str = "Transformer"

        return PredictionResponse(
            player_id=request.player_id,
            player_name=request.player_name,
            predictions=final_prediction,
            model_used=model_used_str,
            confidence="medium"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    info = {
        "lstm_loaded": lstm_model is not None,
        "transformer_loaded": transformer_model is not None,
        "device": str(device)
    }

    if lstm_model is not None:
        info["lstm_parameters"] = sum(p.numel() for p in lstm_model.parameters())

    if transformer_model is not None:
        info["transformer_parameters"] = sum(p.numel() for p in transformer_model.parameters())

    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
