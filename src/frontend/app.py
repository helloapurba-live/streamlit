"""
============================================================================
Streamlit Dashboard for Bank Transaction Anomaly Detection
============================================================================

LEARNING OBJECTIVES:
1. Build multi-page Streamlit application with session state management
2. Integrate with FastAPI backend for real-time and batch predictions
3. Create interactive visualizations for fraud detection
4. Implement job polling for asynchronous batch processing
5. Design operator-friendly UI/UX for fraud investigation

THEORY:
--------
A production fraud detection dashboard requires:
- Multi-page navigation (Overview, Predict, Batch, Investigate, Admin)
- Real-time updates (polling for batch job status)
- Interactive visualizations (charts, tables, SHAP explanations)
- Session state management (preserve user inputs across reruns)
- Error handling (API failures, validation errors)
- Responsive design (works on different screen sizes)

Dashboard Pages:
1. 📊 Overview: Key metrics, model performance, alert summary
2. 🔍 Single Prediction: Manual transaction input and real-time scoring
3. 📦 Batch Processing: CSV upload, job tracking, result download
4. 🕵️ Investigation: Customer deep-dive with transaction history
5. ⚙️ Admin: Model management, MLflow links, system health

ARCHITECTURE:
-------------
┌─────────────────────────────────────────────────────────────────┐
│                   Streamlit Frontend (Port 8501)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │ Overview │  │ Predict  │  │  Batch   │  │ Investig.│  ...  │
│  │  Page    │  │  Page    │  │  Page    │  │  Page    │      │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘      │
│       │             │             │             │              │
│       └─────────────┴─────────────┴─────────────┘              │
│                        │                                        │
│         ┌──────────────▼──────────────┐                        │
│         │  REST API Client (requests) │                        │
│         └──────────────┬──────────────┘                        │
└────────────────────────┼─────────────────────────────────────────┘
                         │
                         │ HTTP Requests
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                         │
│  /predict_single, /predict_batch, /job_status, /download_result │
└───────────────────────────────────────────────────────────────────┘
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import io
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .fraud-alert {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .safe-card {
        background-color: #d1e7dd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #198754;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# API CLIENT FUNCTIONS
# ============================================================================

def check_api_health() -> Dict[str, Any]:
    """Check if backend API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "healthy", "data": response.json()}
        else:
            return {"status": "unhealthy", "error": f"Status code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "error": str(e)}


def predict_single_transaction(transaction_data: Dict) -> Optional[Dict]:
    """Call /predict_single endpoint."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_single",
            json=transaction_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def submit_batch_job(file_bytes: bytes, filename: str) -> Optional[Dict]:
    """Call /predict_batch endpoint."""
    try:
        files = {'file': (filename, file_bytes, 'text/csv')}
        response = requests.post(
            f"{API_BASE_URL}/predict_batch",
            files=files,
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def get_job_status(job_id: str) -> Optional[Dict]:
    """Call /job_status endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/job_status/{job_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def download_batch_results(job_id: str) -> Optional[bytes]:
    """Call /download_result endpoint."""
    try:
        response = requests.get(f"{API_BASE_URL}/download_result/{job_id}", timeout=30)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


def investigate_customer(customer_id: str, query: str) -> Optional[Dict]:
    """Call /agent_investigate endpoint."""
    try:
        payload = {
            "customer_id": customer_id,
            "query": query
        }
        response = requests.post(
            f"{API_BASE_URL}/agent_investigate",
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_gauge_chart(value: float, title: str, color: str = "blue") -> go.Figure:
    """Create a gauge chart for metrics."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    fig.update_layout(height=250)
    return fig


def create_time_series_chart(df: pd.DataFrame) -> go.Figure:
    """Create time series chart of fraud detection over time."""
    fig = px.line(
        df,
        x='timestamp',
        y='fraud_count',
        title='Fraud Detections Over Time',
        labels={'fraud_count': 'Number of Frauds', 'timestamp': 'Time'}
    )
    fig.update_traces(line_color='#dc3545', line_width=3)
    fig.update_layout(height=300)
    return fig


def format_currency(amount: float) -> str:
    """Format amount as currency."""
    return f"${amount:,.2f}"


def load_sample_data() -> pd.DataFrame:
    """Load sample transaction data for demo purposes."""
    # This would normally load from the API or database
    # For demo, create synthetic data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(n)],
        'customer_id': [f'CUST_{np.random.randint(1, 1000):06d}' for _ in range(n)],
        'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='H'),
        'amount': np.random.lognormal(mean=4, sigma=1, size=n),
        'merchant_category': np.random.choice(
            ['grocery', 'gas_transport', 'restaurant', 'online_shopping'], n
        ),
        'is_fraud': np.random.choice([0, 1], n, p=[0.97, 0.03]),
        'fraud_probability': np.random.beta(2, 5, n)
    })

    return df


# ============================================================================
# PAGE: OVERVIEW (Dashboard Home)
# ============================================================================

def page_overview():
    """
    Overview page with key metrics and alerts.

    FEATURES:
    - Real-time fraud rate gauge
    - Transaction volume metrics
    - Recent alerts table
    - Time series fraud detection chart
    - System health status
    """
    st.markdown("<h1 class='main-header'>🏦 Fraud Detection Dashboard</h1>", unsafe_allow_html=True)

    # Check API health
    health = check_api_health()
    if health['status'] == 'healthy':
        st.success("✅ Backend API: Healthy")
    else:
        st.error(f"❌ Backend API: {health.get('error', 'Unhealthy')}")
        st.stop()

    # Load sample data for dashboard
    df = load_sample_data()

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_transactions = len(df)
        st.metric(
            label="📊 Total Transactions (24h)",
            value=f"{total_transactions:,}",
            delta=f"+{np.random.randint(50, 200)}"
        )

    with col2:
        fraud_count = df['is_fraud'].sum()
        st.metric(
            label="🚨 Fraud Detected",
            value=f"{fraud_count}",
            delta=f"+{np.random.randint(1, 5)}",
            delta_color="inverse"
        )

    with col3:
        fraud_rate = fraud_count / total_transactions
        st.metric(
            label="📈 Fraud Rate",
            value=f"{fraud_rate*100:.2f}%",
            delta=f"{np.random.uniform(-0.5, 0.5):.2f}%"
        )

    with col4:
        total_amount = df[df['is_fraud'] == 1]['amount'].sum()
        st.metric(
            label="💰 Fraud Amount Blocked",
            value=format_currency(total_amount),
            delta=format_currency(np.random.uniform(500, 2000))
        )

    st.markdown("---")

    # Fraud rate gauge and time series
    col1, col2 = st.columns([1, 2])

    with col1:
        st.plotly_chart(
            create_gauge_chart(fraud_rate, "Current Fraud Rate", "red"),
            use_container_width=True
        )

    with col2:
        # Create time series data
        hourly_fraud = df.groupby(df['timestamp'].dt.floor('H'))['is_fraud'].sum().reset_index()
        hourly_fraud.columns = ['timestamp', 'fraud_count']
        st.plotly_chart(
            create_time_series_chart(hourly_fraud),
            use_container_width=True
        )

    st.markdown("---")

    # Recent high-risk transactions
    st.subheader("🚨 Recent High-Risk Transactions")

    high_risk = df[df['fraud_probability'] > 0.7].sort_values('timestamp', ascending=False).head(10)

    if len(high_risk) > 0:
        # Format for display
        display_df = high_risk[[
            'transaction_id', 'customer_id', 'timestamp', 'amount',
            'merchant_category', 'fraud_probability'
        ]].copy()

        display_df['amount'] = display_df['amount'].apply(format_currency)
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x*100:.1f}%")
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No high-risk transactions in the selected period.")

    # Model performance summary
    st.markdown("---")
    st.subheader("📊 Model Performance (Last 7 Days)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Precision", "94.2%", "+1.2%")
    with col2:
        st.metric("Recall", "87.5%", "-0.5%")
    with col3:
        st.metric("F1 Score", "90.7%", "+0.3%")


# ============================================================================
# PAGE: SINGLE PREDICTION
# ============================================================================

def page_single_prediction():
    """
    Single transaction prediction page.

    FEATURES:
    - Manual input form for transaction features
    - Real-time prediction on submit
    - SHAP explanation visualization
    - Fraud/legitimate result with confidence
    """
    st.markdown("<h1 class='main-header'>🔍 Single Transaction Prediction</h1>", unsafe_allow_html=True)

    st.markdown("""
    Enter transaction details below to get a real-time fraud prediction.
    The model will analyze the transaction and provide an explanation for its decision.
    """)

    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            customer_id = st.text_input(
                "Customer ID",
                value="CUST_000123",
                help="Enter customer identifier"
            )

            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                value=150.00,
                step=10.0,
                help="Amount in USD"
            )

            merchant_category = st.selectbox(
                "Merchant Category",
                options=[
                    'grocery', 'gas_transport', 'restaurant', 'entertainment',
                    'online_shopping', 'bills_utilities', 'health_fitness', 'travel'
                ],
                index=4
            )

            merchant_id = st.text_input(
                "Merchant ID",
                value="MERCH_01234"
            )

            is_online = st.checkbox("Online Transaction", value=True)

        with col2:
            latitude = st.number_input(
                "Latitude",
                min_value=-90.0,
                max_value=90.0,
                value=40.7128,
                step=0.0001,
                format="%.4f"
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-180.0,
                max_value=180.0,
                value=-74.0060,
                step=0.0001,
                format="%.4f"
            )

            hour = st.slider("Hour of Day", 0, 23, 14)

            day_of_week = st.selectbox(
                "Day of Week",
                options=[
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday"
                ],
                index=2
            )

            is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
            is_night = 1 if hour < 6 else 0

        submitted = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)

    # Process prediction
    if submitted:
        # Prepare request payload
        transaction_data = {
            "customer_id": customer_id,
            "amount": amount,
            "merchant_category": merchant_category,
            "merchant_id": merchant_id,
            "latitude": latitude,
            "longitude": longitude,
            "is_online": 1 if is_online else 0,
            "hour": hour,
            "day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day_of_week),
            "is_weekend": is_weekend,
            "is_night": is_night
        }

        with st.spinner("Analyzing transaction..."):
            result = predict_single_transaction(transaction_data)

        if result:
            st.markdown("---")
            st.subheader("📊 Prediction Result")

            # Display result in card
            fraud_prob = result['fraud_probability']
            is_fraud = result['is_fraud_predicted']

            if is_fraud:
                st.markdown(
                    f"""
                    <div class='fraud-alert'>
                        <h2>🚨 FRAUD ALERT</h2>
                        <p><strong>Fraud Probability:</strong> {fraud_prob*100:.1f}%</p>
                        <p><strong>Transaction ID:</strong> {result['transaction_id']}</p>
                        <p><strong>Model Version:</strong> {result['model_version']}</p>
                        <p><strong>Processing Time:</strong> {result['processing_time_ms']:.1f}ms</p>
                        <p>⚠️ <strong>Recommended Action:</strong> BLOCK or REVIEW</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class='safe-card'>
                        <h2>✅ LEGITIMATE TRANSACTION</h2>
                        <p><strong>Fraud Probability:</strong> {fraud_prob*100:.1f}%</p>
                        <p><strong>Transaction ID:</strong> {result['transaction_id']}</p>
                        <p><strong>Model Version:</strong> {result['model_version']}</p>
                        <p><strong>Processing Time:</strong> {result['processing_time_ms']:.1f}ms</p>
                        <p>✓ <strong>Recommended Action:</strong> APPROVE</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Display explanation
            st.markdown("---")
            st.subheader("🔍 Model Explanation")

            explanation = result.get('explanation', {})

            if explanation.get('method') == 'shap':
                st.markdown("**SHAP Feature Contributions:**")

                # Create DataFrame for visualization
                features_data = explanation.get('top_features', [])
                if features_data:
                    exp_df = pd.DataFrame(features_data)

                    # Create horizontal bar chart
                    fig = go.Figure(go.Bar(
                        x=exp_df['shap_value'],
                        y=exp_df['feature'],
                        orientation='h',
                        marker=dict(
                            color=exp_df['shap_value'],
                            colorscale='RdYlGn_r',
                            showscale=True
                        ),
                        text=exp_df['shap_value'].round(3),
                        textposition='auto'
                    ))
                    fig.update_layout(
                        title="Top Feature Contributions (SHAP Values)",
                        xaxis_title="SHAP Value (→ Fraud)",
                        yaxis_title="Feature",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show feature values
                    st.markdown("**Feature Values:**")
                    for feat in features_data:
                        st.text(f"  • {feat['feature']}: {feat['feature_value']:.2f}")
            else:
                st.info(f"Explanation method: {explanation.get('method', 'none')}")


# ============================================================================
# PAGE: BATCH PROCESSING
# ============================================================================

def page_batch_processing():
    """
    Batch prediction page with job tracking.

    FEATURES:
    - CSV file upload
    - Job submission and tracking
    - Progress bar with polling
    - Result download
    """
    st.markdown("<h1 class='main-header'>📦 Batch Transaction Processing</h1>", unsafe_allow_html=True)

    st.markdown("""
    Upload a CSV file containing multiple transactions for batch processing.
    You can track the job progress and download results when complete.
    """)

    # File upload section
    st.subheader("📤 Upload Transactions")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="CSV must contain columns: customer_id, amount, merchant_category, merchant_id, latitude, longitude, is_online, hour, day_of_week, is_weekend, is_night"
    )

    if uploaded_file is not None:
        # Preview uploaded file
        df_preview = pd.read_csv(uploaded_file)
        st.write(f"📊 Preview ({len(df_preview)} rows):")
        st.dataframe(df_preview.head(10), use_container_width=True)

        # Reset file pointer
        uploaded_file.seek(0)

        if st.button("🚀 Submit Batch Job", use_container_width=True):
            file_bytes = uploaded_file.getvalue()

            with st.spinner("Submitting batch job..."):
                result = submit_batch_job(file_bytes, uploaded_file.name)

            if result:
                st.success(f"✅ Job submitted successfully!")
                st.session_state['current_job_id'] = result['job_id']
                st.rerun()

    # Job tracking section
    st.markdown("---")
    st.subheader("📊 Job Status")

    # Input for job ID (or use current job)
    col1, col2 = st.columns([3, 1])

    with col1:
        job_id_input = st.text_input(
            "Job ID",
            value=st.session_state.get('current_job_id', ''),
            help="Enter job ID to track"
        )

    with col2:
        track_button = st.button("🔍 Track Job", use_container_width=True)

    if track_button and job_id_input:
        st.session_state['tracking_job_id'] = job_id_input

    # Poll job status if tracking
    if 'tracking_job_id' in st.session_state and st.session_state['tracking_job_id']:
        job_id = st.session_state['tracking_job_id']

        # Create placeholder for real-time updates
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        details_placeholder = st.empty()

        # Poll until complete
        max_polls = 60  # Maximum 60 polls (5 minutes with 5-second intervals)
        poll_count = 0

        while poll_count < max_polls:
            status_data = get_job_status(job_id)

            if status_data:
                status = status_data['status']
                progress = status_data.get('progress', 0)

                # Update status
                if status == 'pending':
                    status_placeholder.info(f"⏳ Job Status: **Pending**")
                elif status == 'running':
                    status_placeholder.info(f"⚙️ Job Status: **Running**")
                elif status == 'completed':
                    status_placeholder.success(f"✅ Job Status: **Completed**")
                elif status == 'failed':
                    status_placeholder.error(f"❌ Job Status: **Failed**")
                    error_msg = status_data.get('error_message', 'Unknown error')
                    st.error(f"Error: {error_msg}")
                    break

                # Update progress bar
                progress_placeholder.progress(progress / 100)

                # Show details
                with details_placeholder.container():
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Records", status_data.get('total_records', 'N/A'))
                    with col2:
                        st.metric("Processed", status_data.get('processed_records', 0))
                    with col3:
                        st.metric("Progress", f"{progress:.1f}%")

                # If completed, offer download
                if status == 'completed':
                    st.success("🎉 Batch processing completed!")

                    if st.button("⬇️ Download Results", use_container_width=True):
                        result_bytes = download_batch_results(job_id)
                        if result_bytes:
                            st.download_button(
                                label="💾 Save Results CSV",
                                data=result_bytes,
                                file_name=f"{job_id}_results.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                            # Preview results
                            result_df = pd.read_csv(io.BytesIO(result_bytes))
                            st.subheader("📊 Results Preview")
                            st.dataframe(result_df.head(20), use_container_width=True)

                            # Summary statistics
                            st.subheader("📈 Summary Statistics")
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Total Transactions", len(result_df))
                            with col2:
                                fraud_count = result_df['is_fraud_predicted'].sum()
                                st.metric("Fraud Detected", fraud_count)
                            with col3:
                                fraud_rate = (fraud_count / len(result_df)) * 100
                                st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                    break

                # Exit if job finished or failed
                if status in ['completed', 'failed']:
                    break

            else:
                status_placeholder.error("❌ Failed to fetch job status")
                break

            # Wait before next poll (only if not completed)
            if status_data and status_data['status'] not in ['completed', 'failed']:
                time.sleep(5)
                poll_count += 1
            else:
                break


# ============================================================================
# PAGE: INVESTIGATION
# ============================================================================

def page_investigation():
    """
    Customer investigation page with LLM agent.

    FEATURES:
    - Customer ID search
    - Transaction history display
    - LLM-powered investigation
    - Risk score and recommendations
    """
    st.markdown("<h1 class='main-header'>🕵️ Customer Investigation</h1>", unsafe_allow_html=True)

    st.markdown("""
    Investigate suspicious customer accounts using AI-powered analysis.
    Enter a customer ID to view transaction history and get automated insights.
    """)

    # Customer search
    col1, col2 = st.columns([3, 1])

    with col1:
        customer_id = st.text_input(
            "Customer ID",
            value="CUST_000123",
            help="Enter customer ID to investigate"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        search_button = st.button("🔍 Investigate", use_container_width=True)

    if search_button and customer_id:
        st.markdown("---")

        # Load transaction history (mock data for demo)
        st.subheader(f"📊 Transaction History: {customer_id}")

        # Generate mock transaction history
        df_history = load_sample_data()
        df_history = df_history[df_history['customer_id'] == customer_id]

        if len(df_history) == 0:
            # Create some mock data
            df_history = pd.DataFrame({
                'transaction_id': [f'TXN_{i:06d}' for i in range(10)],
                'timestamp': pd.date_range(end=datetime.now(), periods=10, freq='D'),
                'amount': np.random.lognormal(4, 1, 10),
                'merchant_category': np.random.choice(['grocery', 'restaurant', 'online_shopping'], 10),
                'is_fraud': np.random.choice([0, 1], 10, p=[0.8, 0.2]),
                'fraud_probability': np.random.beta(2, 5, 10)
            })

        # Display transaction table
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Transactions", len(df_history))
        with col2:
            fraud_count = df_history['is_fraud'].sum()
            st.metric("Fraudulent", fraud_count)
        with col3:
            total_amount = df_history['amount'].sum()
            st.metric("Total Amount", format_currency(total_amount))
        with col4:
            avg_fraud_prob = df_history['fraud_probability'].mean()
            st.metric("Avg Fraud Score", f"{avg_fraud_prob*100:.1f}%")

        # LLM Investigation
        st.markdown("---")
        st.subheader("🤖 AI Investigation")

        investigation_query = st.text_area(
            "Investigation Query",
            value="Why was this customer flagged for fraud? What patterns are suspicious?",
            height=100
        )

        if st.button("🚀 Run AI Investigation", use_container_width=True):
            with st.spinner("Running AI investigation..."):
                investigation_result = investigate_customer(customer_id, investigation_query)

            if investigation_result:
                # Display risk score
                risk_score = investigation_result['risk_score']

                if risk_score > 0.7:
                    risk_color = "red"
                    risk_label = "HIGH RISK"
                elif risk_score > 0.4:
                    risk_color = "orange"
                    risk_label = "MEDIUM RISK"
                else:
                    risk_color = "green"
                    risk_label = "LOW RISK"

                st.markdown(
                    f"""
                    <div style='background-color: {risk_color}20; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid {risk_color};'>
                        <h3>Risk Score: {risk_score*100:.1f}% - {risk_label}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Summary
                st.markdown("### 📝 Summary")
                st.write(investigation_result['summary'])

                # Key findings
                st.markdown("### 🔍 Key Findings")
                for finding in investigation_result['key_findings']:
                    st.markdown(f"- {finding}")

                # Recommended actions
                st.markdown("### ✅ Recommended Actions")
                for action in investigation_result['recommended_actions']:
                    st.markdown(f"- {action}")


# ============================================================================
# PAGE: ADMIN
# ============================================================================

def page_admin():
    """
    Admin page for model management and system monitoring.

    FEATURES:
    - Model information and versioning
    - MLflow link
    - System health metrics
    - Model reload functionality
    """
    st.markdown("<h1 class='main-header'>⚙️ Administration</h1>", unsafe_allow_html=True)

    # System health
    st.subheader("🏥 System Health")

    health = check_api_health()

    col1, col2, col3 = st.columns(3)

    with col1:
        if health['status'] == 'healthy':
            st.success("✅ Backend API: Healthy")
        else:
            st.error(f"❌ Backend API: {health.get('error', 'Error')}")

    with col2:
        st.success("✅ Streamlit: Running")

    with col3:
        # Check if MLflow is accessible (port 5000)
        try:
            mlflow_response = requests.get("http://localhost:5000", timeout=2)
            st.success("✅ MLflow: Running")
        except:
            st.warning("⚠️ MLflow: Not accessible")

    # Links
    st.markdown("---")
    st.subheader("🔗 Quick Links")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📚 FastAPI Docs")
        st.markdown("[Open API Documentation](http://localhost:8000/docs)")

    with col2:
        st.markdown("### 🧪 MLflow UI")
        st.markdown("[Open MLflow](http://localhost:5000)")

    with col3:
        st.markdown("### 📊 Prometheus Metrics")
        st.markdown("[View Metrics](http://localhost:8000/metrics)")

    # Model information
    st.markdown("---")
    st.subheader("🤖 Model Information")

    if health['status'] == 'healthy':
        models_available = health['data'].get('models_available', [])

        st.write(f"**Models Loaded:** {len(models_available)}")

        for model_name in models_available:
            with st.expander(f"📦 {model_name}"):
                st.markdown(f"""
                - **Model Name:** {model_name}
                - **Version:** 1.0
                - **Type:** {'Random Forest' if 'sklearn' in model_name else 'PyTorch MLP'}
                - **Status:** ✅ Loaded
                - **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}
                """)

    # Configuration
    st.markdown("---")
    st.subheader("⚙️ Configuration")

    with st.expander("🔧 Settings"):
        st.text(f"API Base URL: {API_BASE_URL}")
        st.text(f"Polling Interval: 5 seconds")
        st.text(f"Max Batch Size: 10,000 rows")
        st.text(f"Timeout: 30 seconds")

    # Danger zone
    st.markdown("---")
    st.subheader("⚠️ Danger Zone")

    if st.button("🔄 Reload Models", use_container_width=True):
        st.warning("Model reload would be triggered here (not implemented in demo)")

    if st.button("🗑️ Clear Job History", use_container_width=True):
        st.warning("Job history clearing would happen here (not implemented in demo)")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Main application with sidebar navigation.

    Pages:
    1. Overview (Dashboard)
    2. Single Prediction
    3. Batch Processing
    4. Investigation
    5. Admin
    """
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")

    page = st.sidebar.radio(
        "Select Page",
        options=[
            "📊 Overview",
            "🔍 Single Prediction",
            "📦 Batch Processing",
            "🕵️ Investigation",
            "⚙️ Admin"
        ],
        index=0
    )

    # Info section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.markdown("""
    **Fraud Detection Dashboard**

    Real-time and batch fraud detection
    powered by ML/DL models.

    **Tech Stack:**
    - Frontend: Streamlit
    - Backend: FastAPI
    - ML: PyTorch + Scikit-Learn
    - Optimization: Optuna
    - Tracking: MLflow
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Route to appropriate page
    if page == "📊 Overview":
        page_overview()
    elif page == "🔍 Single Prediction":
        page_single_prediction()
    elif page == "📦 Batch Processing":
        page_batch_processing()
    elif page == "🕵️ Investigation":
        page_investigation()
    elif page == "⚙️ Admin":
        page_admin()


if __name__ == "__main__":
    main()


"""
============================================================================
LINE-BY-LINE EXPLANATION
============================================================================

Lines 1-72: Module docstring with learning objectives, theory, and architecture
    - Explains multi-page Streamlit design
    - Shows integration with FastAPI backend
    - Describes session state and polling patterns

Lines 74-82: Imports
    - streamlit: web framework
    - requests: HTTP client for API calls
    - plotly: interactive charts
    - pandas/numpy: data manipulation

Lines 88: API configuration (FastAPI backend URL)

Lines 90-98: Streamlit page configuration
    - Wide layout for dashboard
    - Custom page title and icon
    - Expanded sidebar by default

Lines 100-138: Custom CSS styling
    - Card styles for metrics and alerts
    - Color coding (red=fraud, green=safe, yellow=warning)
    - Responsive button styling

Lines 145-169: API client helper functions
    - check_api_health: verify backend is running
    - predict_single_transaction: call /predict_single
    - submit_batch_job: upload CSV for batch processing
    - get_job_status: poll job progress
    - download_batch_results: retrieve completed batch results
    - investigate_customer: call LLM agent endpoint

Lines 176-197: Utility functions for visualization
    - create_gauge_chart: circular gauge for fraud rate
    - create_time_series_chart: line chart for trends
    - format_currency: display amounts as $X,XXX.XX
    - load_sample_data: generate demo data

Lines 203-310: page_overview function (Dashboard home)
    - Shows key metrics in columns (total transactions, fraud count, fraud rate)
    - Displays fraud rate gauge chart
    - Time series of fraud detections
    - Recent high-risk transactions table
    - Model performance metrics

Lines 317-497: page_single_prediction function
    - Input form with two columns
    - Customer ID, amount, merchant, location fields
    - Hour, day of week, online/offline toggle
    - Submits to /predict_single endpoint
    - Displays result in colored card (red=fraud, green=safe)
    - Shows SHAP explanation with horizontal bar chart

Lines 504-678: page_batch_processing function
    - CSV file uploader with preview
    - Submit batch job button
    - Job tracking with polling loop
    - Progress bar updates every 5 seconds
    - Download results button when complete
    - Summary statistics on results

Lines 685-823: page_investigation function
    - Customer ID search input
    - Transaction history table (mock data)
    - Summary metrics (total transactions, fraud count, average score)
    - LLM investigation query input
    - Calls /agent_investigate endpoint
    - Displays risk score, findings, and recommended actions

Lines 830-920: page_admin function
    - System health checks (API, Streamlit, MLflow)
    - Quick links to FastAPI docs, MLflow UI, metrics
    - Model information display
    - Configuration settings
    - Danger zone actions (reload models, clear jobs)

Lines 927-978: main function (Application entry point)
    - Sidebar navigation with radio buttons
    - Routes to appropriate page function
    - Sidebar info with tech stack
    - Last updated timestamp

Lines 981-983: Main execution block

============================================================================
SAMPLE UI FLOW
============================================================================

SCENARIO: Operator investigates suspicious high-value transaction

1. User navigates to "📊 Overview" page
   - Sees fraud rate gauge at 3.2% (yellow zone)
   - Notices high-risk transaction: CUST_005432, $1,250 online purchase
   - Clicks on customer link

2. User switches to "🔍 Single Prediction" page
   - Enters transaction details:
       * Amount: $1,250
       * Merchant: online_shopping
       * Location: 40.7128, -74.0060 (NYC)
       * Hour: 23 (11 PM)
       * Weekend: Yes, Night: Yes
   - Clicks "Analyze Transaction"

3. System calls FastAPI /predict_single
   - Returns fraud_probability: 0.82
   - Status: FRAUD ALERT (red card)
   - SHAP explanation shows:
       * Amount (+0.25): High value
       * is_night (+0.12): Unusual time
       * is_online (+0.08): Online risk

4. User switches to "🕵️ Investigation" page
   - Enters customer ID: CUST_005432
   - Clicks "Investigate"
   - Views transaction history: 8 transactions in 24 hours (velocity anomaly)
   - Runs AI investigation
   - AI response:
       * Risk Score: 75% (HIGH RISK)
       * Findings: Multiple high-value transactions, new merchant categories
       * Actions: Contact customer, place hold, review with fraud team

5. Operator takes action: Blocks transaction and flags account

============================================================================
SAMPLE INPUT (Single Prediction Form)
============================================================================

Customer ID: CUST_000987
Amount: $850.00
Merchant Category: online_shopping
Merchant ID: MERCH_07654
Latitude: 40.7128
Longitude: -74.0060
Is Online: Yes (checked)
Hour: 23
Day of Week: Saturday
Is Weekend: Yes (auto-calculated)
Is Night: Yes (auto-calculated)

============================================================================
SAMPLE OUTPUT (Prediction Result)
============================================================================

🚨 FRAUD ALERT

Fraud Probability: 78.5%
Transaction ID: 8f3a2b1c-4d5e-6f7g-8h9i-0j1k2l3m4n5o
Model Version: sklearn_rf
Processing Time: 15.3ms

⚠️ Recommended Action: BLOCK or REVIEW

---

🔍 Model Explanation

SHAP Feature Contributions:

[Bar chart showing:]
amount          ████████████ 0.250
is_night        ██████ 0.120
is_online       ████ 0.080
hour            ██ 0.045
merchant_risk   █ 0.030

Feature Values:
  • amount: 850.00
  • is_night: 1.00
  • is_online: 1.00
  • hour: 23.00
  • merchant_risk: 0.65

============================================================================
POWERSHELL COMMANDS TO RUN
============================================================================

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Ensure backend is running first
# (In separate terminal: uvicorn src.backend.app:app --reload)

# Run Streamlit frontend
cd C:\Users\YourName\ml-fraud-dashboard
streamlit run src\frontend\app.py --server.port 8501

# Expected output:
#
#  You can now view your Streamlit app in your browser.
#
#  Local URL: http://localhost:8501
#  Network URL: http://192.168.1.100:8501
#
# Open browser automatically to http://localhost:8501

# Test in browser:
# 1. Navigate to http://localhost:8501
# 2. Should see dashboard with "Backend API: Healthy" message
# 3. Try single prediction page
# 4. Try batch upload with sample CSV

============================================================================
EXERCISE
============================================================================

TASK: Add a new page called "📈 Analytics" that shows:
      1. Fraud detection trends over time (daily aggregation)
      2. Top merchants by fraud count
      3. Geographic heatmap of fraud locations
      4. Category-wise fraud distribution

REQUIREMENTS:
1. Add new page to sidebar navigation
2. Create page_analytics() function
3. Use plotly for interactive charts
4. Include date range selector
5. Add export to PDF functionality (optional)

SOLUTION OUTLINE:
-----------------

def page_analytics():
    st.markdown("<h1 class='main-header'>📈 Analytics Dashboard</h1>", unsafe_allow_html=True)

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Load data (filtered by date range)
    df = load_sample_data()
    df_filtered = df[(df['timestamp'] >= pd.Timestamp(start_date)) &
                     (df['timestamp'] <= pd.Timestamp(end_date))]

    # 1. Trend chart
    st.subheader("📊 Fraud Detection Trends")
    daily_fraud = df_filtered.groupby(df_filtered['timestamp'].dt.date)['is_fraud'].sum()
    fig = px.line(x=daily_fraud.index, y=daily_fraud.values,
                  labels={'x': 'Date', 'y': 'Fraud Count'})
    st.plotly_chart(fig, use_container_width=True)

    # 2. Top merchants
    st.subheader("🏪 Top Merchants by Fraud")
    merchant_fraud = df_filtered[df_filtered['is_fraud']==1].groupby('merchant_category').size()
    fig = px.bar(x=merchant_fraud.index, y=merchant_fraud.values)
    st.plotly_chart(fig, use_container_width=True)

    # 3. Geographic heatmap (would use plotly scatter_mapbox in production)
    st.subheader("🗺️ Fraud Geographic Distribution")
    st.info("Heatmap would appear here with plotly.express.scatter_mapbox")

    # 4. Category distribution
    st.subheader("📊 Category-wise Fraud Distribution")
    category_dist = df_filtered.groupby('merchant_category')['is_fraud'].mean() * 100
    fig = px.pie(values=category_dist.values, names=category_dist.index)
    st.plotly_chart(fig, use_container_width=True)

Then add to main():
    page = st.sidebar.radio(
        "Select Page",
        options=[
            "📊 Overview",
            "🔍 Single Prediction",
            "📦 Batch Processing",
            "📈 Analytics",  # ← NEW
            "🕵️ Investigation",
            "⚙️ Admin"
        ]
    )

    if page == "📈 Analytics":
        page_analytics()

============================================================================
"""
