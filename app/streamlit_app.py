"""
NBA Insights - Streamlit Demo
Pick a player and see what our models predict for their next game
"""

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os
import sys

# Add paths so imports work
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models', 'transformer'))

from lstm_model import PlayerLSTMWithAttention
from transformer_model import PlayerTransformerClassToken

# Page setup
st.set_page_config(
    page_title="NBA Insights - AI Player Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Some custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .model-comparison {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 15px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Cache the models so we don't reload them every time
@st.cache_resource
def load_models():
    """Load our trained models"""
    device = torch.device('cpu')

    base_path = os.path.dirname(os.path.dirname(__file__))

    # Figure out how many features we have
    feature_cols_path = os.path.join(base_path, 'data', 'processed', 'feature_columns.txt')
    with open(feature_cols_path, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    target_cols_path = os.path.join(base_path, 'data', 'processed', 'target_columns.txt')
    with open(target_cols_path, 'r') as f:
        target_cols = [line.strip() for line in f.readlines()]

    input_size = len(feature_cols)
    output_size = len(target_cols)

    # Load LSTM
    lstm_model = PlayerLSTMWithAttention(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_size=output_size
    )
    lstm_path = os.path.join(base_path, 'models', 'saved', 'lstm_best.pth')
    lstm_model.load_state_dict(torch.load(lstm_path, weights_only=False, map_location=device)['model_state_dict'])
    lstm_model.eval()

    # Load Transformer
    transformer_model = PlayerTransformerClassToken(
        input_size=input_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dropout=0.3,
        output_size=output_size
    )
    transformer_path = os.path.join(base_path, 'models', 'saved', 'transformer_best.pth')
    transformer_model.load_state_dict(torch.load(transformer_path, weights_only=False, map_location=device)['model_state_dict'])
    transformer_model.eval()

    # Load scaler
    scaler_path = os.path.join(base_path, 'models', 'saved', 'demo_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return lstm_model, transformer_model, scaler, feature_cols, target_cols

@st.cache_data
def load_player_data():
    """Load player data for the dropdown"""
    base_path = os.path.dirname(os.path.dirname(__file__))
    df = pd.read_csv(
        os.path.join(base_path, 'data', 'processed', 'player_features_final.csv'),
        parse_dates=['GAME_DATE']
    )
    return df

def get_player_sequence(df, player_name, feature_cols, sequence_length=10):
    """Get a player's recent games for prediction"""
    player_df = df[df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')

    if len(player_df) < sequence_length:
        return None, None

    # Grab their last 10 games
    recent_games = player_df.tail(sequence_length)
    sequence = recent_games[feature_cols].values.astype(np.float64)

    # Some info for display
    latest_game = player_df.iloc[-1]
    player_info = {
        'name': player_name,
        'team': latest_game.get('TEAM_ABBREVIATION', 'N/A'),
        'last_game': latest_game['GAME_DATE'],
        'avg_pts': player_df['TARGET_PTS'].mean(),
        'avg_reb': player_df['TARGET_REB'].mean(),
        'avg_ast': player_df['TARGET_AST'].mean(),
    }

    return sequence, player_info

def make_prediction(model, sequence, scaler):
    """Run the model and get a prediction"""
    # Normalize
    seq_flat = sequence.reshape(-1, sequence.shape[-1])
    seq_normalized = scaler.transform(seq_flat)
    seq_normalized = seq_normalized.reshape(1, sequence.shape[0], sequence.shape[1])

    seq_tensor = torch.FloatTensor(seq_normalized)

    with torch.no_grad():
        prediction = model(seq_tensor)

    return prediction.numpy()[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">🏀 NBA Insights</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem;">AI-Powered Player Performance Predictions</p>', unsafe_allow_html=True)
    st.markdown('---')

    # Try to load everything
    try:
        lstm_model, transformer_model, scaler, feature_cols, target_cols = load_models()
        df = load_player_data()
        models_loaded = True
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        models_loaded = False

    if not models_loaded:
        st.stop()

    # Sidebar for player selection
    st.sidebar.header("Player Selection")

    # Sort players by how many games they have (stars first)
    player_games = df.groupby('PLAYER_NAME').size().sort_values(ascending=False)
    players = player_games.index.tolist()

    selected_player = st.sidebar.selectbox(
        "Select a Player",
        players,
        index=0
    )

    # Model selection
    st.sidebar.header("Model Settings")
    model_choice = st.sidebar.radio(
        "Prediction Model",
        ["Ensemble (LSTM + Transformer)", "LSTM Only", "Transformer Only"],
        index=0
    )

    # About section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        """
        **NBA Insights** uses deep learning to predict player performance:

        - **LSTM**: Captures sequential patterns in game history
        - **Transformer**: Uses attention for contextual relationships
        - **Features**: 50+ engineered features including rolling averages, rest days, opponent strength
        """
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header(f"Predictions for {selected_player}")

        sequence, player_info = get_player_sequence(df, selected_player, feature_cols)

        if sequence is None:
            st.warning("Not enough game history for this player (need at least 10 games)")
            st.stop()

        # Get predictions from both models
        lstm_pred = make_prediction(lstm_model, sequence, scaler)
        transformer_pred = make_prediction(transformer_model, sequence, scaler)

        # Average them for ensemble
        ensemble_pred = (lstm_pred + transformer_pred) / 2

        # Pick which one to display
        if model_choice == "Ensemble (LSTM + Transformer)":
            pred = ensemble_pred
            model_name = "Ensemble"
        elif model_choice == "LSTM Only":
            pred = lstm_pred
            model_name = "LSTM"
        else:
            pred = transformer_pred
            model_name = "Transformer"

        # Show the predictions
        st.subheader(f"Next Game Predictions ({model_name})")

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.metric(
                label="Points",
                value=f"{pred[0]:.1f}",
                delta=f"{pred[0] - player_info['avg_pts']:.1f} vs avg"
            )

        with metric_cols[1]:
            st.metric(
                label="Rebounds",
                value=f"{pred[1]:.1f}",
                delta=f"{pred[1] - player_info['avg_reb']:.1f} vs avg"
            )

        with metric_cols[2]:
            st.metric(
                label="Assists",
                value=f"{pred[2]:.1f}",
                delta=f"{pred[2] - player_info['avg_ast']:.1f} vs avg"
            )

        with metric_cols[3]:
            st.metric(
                label="Fantasy Points",
                value=f"{pred[3]:.1f}",
                delta=None
            )

        # Compare models
        st.markdown("---")
        st.subheader("Model Comparison")

        comparison_df = pd.DataFrame({
            'Metric': ['Points', 'Rebounds', 'Assists', 'Fantasy Pts'],
            'LSTM': lstm_pred,
            'Transformer': transformer_pred,
            'Ensemble': ensemble_pred,
            'Season Avg': [player_info['avg_pts'], player_info['avg_reb'], player_info['avg_ast'], np.nan]
        })

        st.dataframe(
            comparison_df.style.format({
                'LSTM': '{:.1f}',
                'Transformer': '{:.1f}',
                'Ensemble': '{:.1f}',
                'Season Avg': '{:.1f}'
            }),
            use_container_width=True
        )

        # Bar chart
        chart_data = pd.DataFrame({
            'LSTM': lstm_pred[:3],
            'Transformer': transformer_pred[:3],
            'Ensemble': ensemble_pred[:3]
        }, index=['Points', 'Rebounds', 'Assists'])

        st.bar_chart(chart_data)

    with col2:
        st.header("Player Info")

        st.markdown(f"""
        <div class="prediction-card">
            <h3>{player_info['name']}</h3>
            <p><strong>Last Game:</strong> {player_info['last_game'].strftime('%Y-%m-%d')}</p>
            <hr>
            <p><strong>Season Averages:</strong></p>
            <p>Points: {player_info['avg_pts']:.1f}</p>
            <p>Rebounds: {player_info['avg_reb']:.1f}</p>
            <p>Assists: {player_info['avg_ast']:.1f}</p>
        </div>
        """, unsafe_allow_html=True)

        # Recent games
        st.subheader("Recent Games")

        player_df = df[df['PLAYER_NAME'] == selected_player].sort_values('GAME_DATE', ascending=False).head(5)

        recent_display = player_df[['GAME_DATE', 'OPPONENT', 'TARGET_PTS', 'TARGET_REB', 'TARGET_AST']].copy()
        recent_display.columns = ['Date', 'Opponent', 'PTS', 'REB', 'AST']
        recent_display['Date'] = recent_display['Date'].dt.strftime('%m/%d')

        st.dataframe(recent_display, use_container_width=True, hide_index=True)

        # Model performance
        st.subheader("Model Performance")
        st.markdown("""
        <div class="model-comparison">
            <h4>Test Set Metrics</h4>
            <table style="width:100%">
                <tr><th></th><th>LSTM</th><th>Transformer</th></tr>
                <tr><td>Points MAE</td><td>2.92</td><td>2.93</td></tr>
                <tr><td>Points R²</td><td>0.737</td><td>0.740</td></tr>
                <tr><td>Rebounds MAE</td><td>1.39</td><td>1.40</td></tr>
                <tr><td>Assists MAE</td><td>1.00</td><td>0.98</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <p style="text-align: center; color: gray;">
        NBA Insights - CECS 458 Deep Learning Term Project<br>
        Team 6: Paul Semaan, Kyler Sison<br>
        California State University, Long Beach
    </p>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
