import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from io import StringIO
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load environment variables for Gemini
load_dotenv()

# Configure page settings
st.set_page_config(page_title="Data Insights Hub", layout="wide")

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page
    st.rerun()

# ==================== PAGE 1: HOME PAGE ====================
def home_page():
    st.title("📊 Data Insights Hub")
    st.markdown("Select an analysis option:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/2489/2489753.png", width=100)
        if st.button("Time Series Forecasting (LSTM)"):
            navigate_to("time_series")
    
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/2489/2489772.png", width=100)
        if st.button("Graphs to Insights (AI)"):
            navigate_to("graphs_insights")
    
    with col3:
        st.image("https://cdn-icons-png.flaticon.com/512/2489/2489718.png", width=100)
        if st.button("Dataset to Graphs"):
            navigate_to("dataset_graphs")

# ==================== PAGE 2: TIME SERIES FORECASTING (LSTM) ====================
def time_series_page():
    st.title("📈 Time Series Forecasting with LSTM")
    
    if st.button("← Back to Home"):
        navigate_to("Home")
    
    st.markdown("""
    Upload your time series data (CSV with 'Date' column) to get predictions.
    The app will automatically use the first numeric column for forecasting.
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="ts_uploader")
    
    if uploaded_file is not None:
        try:
            # Load dataset
            df = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            
            # Sort and clean data
            df = df.sort_index()
            df = df.fillna(method='ffill')

            # Use only the first numeric column for prediction
            target_column = df.select_dtypes(include=[np.number]).columns[0]
            df = df[[target_column]].rename(columns={target_column: "Close"})

            # Display original data
            st.subheader("Original Data")
            st.line_chart(df)
            
            # Scale data
            scaler = MinMaxScaler(feature_range=(0, 1))
            df_scaled = scaler.fit_transform(df)

            # Prepare data for LSTM
            sequence_length = 60
            X, y = [], []
            for i in range(sequence_length, len(df_scaled)):
                X.append(df_scaled[i-sequence_length:i, 0])
                y.append(df_scaled[i, 0])

            X, y = np.array(X), np.array(y)
            X = X.reshape((X.shape[0], X.shape[1], 1))

            # Build & train LSTM model
            with st.spinner('Training the LSTM model...'):
                model = Sequential([
                    LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
                    Dropout(0.2),
                    LSTM(100, return_sequences=False),
                    Dropout(0.2),
                    Dense(50),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mean_squared_error')
                history = model.fit(X, y, epochs=20, batch_size=32, verbose=0)
                
                # Plot training loss
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'])
                ax.set_title('Model Training Loss')
                ax.set_ylabel('Loss')
                ax.set_xlabel('Epoch')
                st.pyplot(fig)

            # Predict future values
            future_days = st.slider("Select number of days to predict:", 7, 90, 30)
            predictions = []
            last_sequence = df_scaled[-sequence_length:].reshape(1, sequence_length, 1)

            with st.spinner(f'Predicting next {future_days} days...'):
                for _ in range(future_days):
                    pred = model.predict(last_sequence, verbose=0)[0]
                    predictions.append(pred)
                    last_sequence = np.append(last_sequence[:, 1:, :], [[pred]], axis=1)

            # Convert predictions back to original scale
            predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

            # Create future dates
            last_date = df.index[-1]
            future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]

            # Combine historical and predicted data
            historical = df[['Close']].rename(columns={'Close': 'Actual'})
            predicted_df = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted'])
            
            # Plot results
            st.subheader("Historical Data vs Predictions")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(historical.index, historical['Actual'], label='Historical')
            ax.plot(predicted_df.index, predicted_df['Predicted'], 'r--', label='Predicted')
            ax.set_title('Time Series Forecasting')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

            # Generate insights
            historical_max = df['Close'].max()
            historical_min = df['Close'].min()
            last_year_price_change = ((df.iloc[-1]['Close'] - df.iloc[-365]['Close']) / df.iloc[-365]['Close']) * 100 if len(df) > 365 else 0
            last_30_days_avg = df.iloc[-30:]['Close'].mean()
            volatility = df['Close'].std()
            price_trend = "rising" if df.iloc[-1]['Close'] > df.iloc[-30]['Close'] else "falling"
            predicted_trend = "increase" if predicted_prices[-1][0] > df.iloc[-1]['Close'] else "decrease"
            predicted_change = ((predicted_prices[-1][0] - df.iloc[-1]['Close']) / df.iloc[-1]['Close']) * 100

            # Display insights in columns
            st.subheader("Key Insights")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Historical Maximum", f"${historical_max:.2f}")
                st.metric("30-Day Average", f"${last_30_days_avg:.2f}")
                st.metric("Current Trend", price_trend.capitalize())
                
            with col2:
                st.metric("Historical Minimum", f"${historical_min:.2f}")
                st.metric("Volatility", f"${volatility:.2f}", 
                         help="Standard deviation of prices")
                st.metric("Predicted Trend", predicted_trend.capitalize())
                
            with col3:
                st.metric("1-Year Change", f"{last_year_price_change:.2f}%",
                         help="Percentage change over last year" if len(df) > 365 else "Insufficient data")
                st.metric("Predicted Change", f"{predicted_change:.2f}%",
                         help=f"Percentage change predicted over {future_days} days")
                st.metric("Final Predicted Value", f"${predicted_prices[-1][0]:.2f}")

            # Show raw predictions
            st.subheader("Prediction Data")
            st.dataframe(predicted_df.style.format("{:.2f}"))
            
            # Download button for predictions
            csv = predicted_df.to_csv().encode('utf-8')
            st.download_button(
                label="Download predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==================== PAGE 3: GRAPHS TO INSIGHTS (GEMINI AI) ====================
def graphs_insights_page():
    st.title("🤖 Graphs to Insights (AI Analysis)")
    
    if st.button("← Back to Home"):
        navigate_to("Home")
    
    # Initialize Gemini if API key is available
    if 'GEMINI_API_KEY' in os.environ:
        genai.configure(api_key=os.environ['GEMINI_API_KEY'])
        model = genai.GenerativeModel("gemini-1.5-flash")
    else:
        st.warning("Gemini API key not found. Some features may be limited.")
        model = None
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of your graph/chart",
        type=["png", "jpg", "jpeg"],
        key="graph_uploader"
    )
    
    # User prompt input
    user_prompt = st.text_input(
        "Ask something about the graph (optional)",
        value="Analyze this graph and explain key insights"
    )
    
    # Process image on button click
    if st.button("Generate Insights") and uploaded_file:
        if model is None:
            st.error("Gemini API not configured. Please set GEMINI_API_KEY in your environment.")
            return
        
        try:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Graph", use_column_width=True)
            
            # Convert image to bytes
            image_bytes = io.BytesIO(uploaded_file.read())
            image_pil = Image.open(image_bytes)
            
            # Ensure RGB mode (required by Gemini)
            if image_pil.mode != "RGB":
                image_pil = image_pil.convert("RGB")
            
            # Prepare image for Gemini
            image_bytes.seek(0)
            encoded_image = {
                "mime_type": "image/jpeg",
                "data": image_bytes.read()
            }
            
            # Generate response using Gemini AI
            with st.spinner("Analyzing graph with AI..."):
                response = model.generate_content([user_prompt, encoded_image])
                response_text = response.text if response.text else "No response generated."
            
            # Display response
            st.subheader("AI Analysis")
            st.write(response_text)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# ==================== PAGE 4: DATASET TO GRAPHS ====================
def dataset_graphs_page():
    st.title("📊 Dataset to Graphs")
    
    if st.button("← Back to Home"):
        navigate_to("Home")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="graph_gen_uploader")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Show raw data
            st.subheader("Raw Data")
            st.dataframe(df.head())
            
            # Get column names
            columns = df.columns.tolist()
            
            # Visualization controls
            st.sidebar.subheader("Chart Configuration")
            
            # Chart type selection
            chart_type = st.sidebar.selectbox(
                "Select Chart Type",
                ["Bar", "Line", "Scatter", "Pie", "Histogram", "Box", "Violin"]
            )
            
            # Initialize figure
            fig = None
            
            # Column selection based on chart type
            if chart_type in ["Bar", "Line", "Scatter"]:
                x_col = st.sidebar.selectbox("X-axis Column", columns)
                y_col = st.sidebar.selectbox("Y-axis Column", columns)
                
                if chart_type == "Bar":
                    fig = px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {x_col} vs {y_col}")
                elif chart_type == "Line":
                    fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {x_col} vs {y_col}")
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
                    
            elif chart_type == "Pie":
                x_col = st.sidebar.selectbox("Category Column", columns)
                y_col = st.sidebar.selectbox("Value Column", columns)
                fig = px.pie(df, names=x_col, values=y_col, title=f"Pie Chart: {x_col}")
                
            elif chart_type in ["Histogram", "Box", "Violin"]:
                x_col = st.sidebar.selectbox("Column", columns)
                
                if chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, title=f"Histogram: {x_col}")
                elif chart_type == "Box":
                    fig = px.box(df, y=x_col, title=f"Box Plot: {x_col}")
                elif chart_type == "Violin":
                    fig = px.violin(df, y=x_col, title=f"Violin Plot: {x_col}")
            
            # Additional customization
            if fig:
                st.sidebar.subheader("Customization")
                
                if st.sidebar.checkbox("Add Color"):
                    color_col = st.sidebar.selectbox("Color by", [None] + columns)
                    if color_col:
                        fig.update_traces(marker=dict(color=color_col))
                
                # Display the chart
                st.subheader("Visualization")
                st.plotly_chart(fig, use_container_width=True)
                
                # Download options
                st.sidebar.subheader("Export Options")
                st.sidebar.download_button(
                    label="Download as HTML",
                    data=fig.to_html(),
                    file_name="chart.html",
                    mime="text/html"
                )
            
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# ==================== MAIN APP ROUTER ====================
if st.session_state.current_page == "Home":
    home_page()
elif st.session_state.current_page == "time_series":
    time_series_page()
elif st.session_state.current_page == "graphs_insights":
    graphs_insights_page()
elif st.session_state.current_page == "dataset_graphs":
    dataset_graphs_page()