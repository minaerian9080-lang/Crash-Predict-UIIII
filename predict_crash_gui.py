import streamlit as st
import pandas as pd
import train  # دالة predict_next_event
import train_rnn

# -----------------------
# Functions for prediction
# -----------------------

def get_model_prediction_crashCNN(placeholder):
    with st.spinner("Predicting with Crash CNN..."):
        result = train.predict_next_event()
        placeholder.subheader(f"Prediction (Crash CNN): {result}")

def get_model_prediction_crashRNN(placeholder):
    with st.spinner("Predicting with Crash RNN..."):
        result = train_rnn.predict_rnn()
        placeholder.subheader(f"Prediction (Crash RNN): {result}")

# -----------------------
# Function to display last game data
# -----------------------
def load_game_data(placeholder):
    try:
        df = pd.read_csv("data.csv")
        last_game = df.iloc[-1]  # آخر صف
    except Exception:
        last_game = None

    with placeholder.container():
        st.subheader("Last Game Data")
        if last_game is not None:
            st.metric("Total Payout", last_game.get("payout", "N/A"))
            st.metric("Game Multiplier", last_game.get("ticket", "N/A")/100 if isinstance(last_game.get("ticket"), (int,float)) else last_game.get("ticket"))
            st.metric("Number of Bets", last_game.get("numberOfBets", "N/A"))
            st.write(f"Game ID: {last_game.get('gameId','N/A')}")
            st.write(f"Server Seed: {last_game.get('serverSeed','N/A')}")
            st.write(f"Started At: {last_game.get('startedAt','N/A')}")
            st.write(f"End Time: {last_game.get('endTime','N/A')}")
        else:
            st.write("No game data available.")

# -----------------------
# Streamlit UI
# -----------------------

st.title("Crash Predictor - Safe Version")

# Select model
selected_model = st.sidebar.selectbox("Choose Model", ["Crash CNN", "Crash RNN"])

# Placeholders
data_placeholder = st.empty()
prediction_placeholder = st.empty()

# Load last game data safely
load_game_data(data_placeholder)

# Predict button
if st.button("Predict Next Game"):
    if selected_model == "Crash CNN":
        get_model_prediction_crashCNN(prediction_placeholder)
    else:
        get_model_prediction_crashRNN(prediction_placeholder)
                
