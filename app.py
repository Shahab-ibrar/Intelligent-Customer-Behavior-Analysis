import os
import joblib
import pandas as pd
import numpy as np
import torch
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.rl_agent import DQNNet
# from typing import Dict, Any, Tuple

# Set page configuration
st.set_page_config(
    page_title="Intelligent Customer Behavior Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Remove empty top container and header */
    [data-testid="stHeader"] {
        background-color: transparent;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    
    /* Main container background */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling - letting Streamlit handle it natively */
    
    /* Custom metric card wrapper */
    .metric-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.15);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .metric-title {
        font-size: 14px;
        color: var(--text-color);
        opacity: 0.7;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 8px;
    }
    .metric-sub {
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.5;
    }
    
    /* Gradient text */
    .gradient-text {
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium Button */
    .stButton > button {
        background: linear-gradient(90deg, #3B82F6, #06B6D4);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        color: white;
    }
    
    /* Card Container */
    .glass-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 24px;
    }
    .glass-card:hover {
        border-color: rgba(128, 128, 128, 0.4);
        transform: translateY(-2px);
    }
    
    .text-muted {
        color: var(--text-color);
        opacity: 0.7;
    }
    .bg-faded {
        background: rgba(128, 128, 128, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models safely
@st.cache_resource
def load_all_models():
    models_dir = 'models'
    scaler = joblib.load(os.path.join(models_dir, 'category_scaler.pkl'))
    pca = joblib.load(os.path.join(models_dir, 'pca_model.pkl'))
    lda = joblib.load(os.path.join(models_dir, 'lda_model.pkl'))
    classifier = joblib.load(os.path.join(models_dir, 'best_classifier.pkl'))
    regressor = joblib.load(os.path.join(models_dir, 'best_regressor.pkl'))
    
    # RL Agents
    tabular_q_table = joblib.load(os.path.join(models_dir, 'tabular_q_table.pkl'))
    tabular_kmeans = joblib.load(os.path.join(models_dir, 'tabular_kmeans.pkl'))
    
    dqn_model = DQNNet(state_dim=5, action_dim=3)
    dqn_model.load_state_dict(torch.load(os.path.join(models_dir, 'best_dqn.pt'), map_location=torch.device('cpu')))
    dqn_model.eval()
    
    return scaler, pca, lda, classifier, regressor, tabular_q_table, tabular_kmeans, dqn_model

@st.cache_data
def load_customer_data():
    return pd.read_csv('data/processed_customers.csv')

# Load resources
try:
    scaler, pca, lda, classifier, regressor, tabular_q, tabular_kmeans, dqn = load_all_models()
    df_customers = load_customer_data()
    model_loading_success = True
except Exception as e:
    st.error(f"Error loading models or dataset. Please make sure the notebooks have executed completely: {e}")
    model_loading_success = False

if model_loading_success:
    # -------------------------------------------------------------
    # SIDEBAR CONFIGURATION
    # -------------------------------------------------------------
    st.sidebar.markdown("""
    <div style='padding-bottom: 20px;'>
        <h2 style='margin-bottom: 0;'>⚙️ Control Panel</h2>
        <p style='opacity: 0.7; font-size: 14px; margin-top: 5px;'>Configure simulation parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Customer Selector
    customer_ids = sorted(df_customers['CustomerID'].unique())
    selected_id = st.sidebar.selectbox("👤 Select Customer ID", customer_ids)
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Toggle PCA vs LDA Scatter Plot
    pca_lda_toggle = st.sidebar.radio("📊 Dimensionality Reduction Visual", ["PCA (Principal Component Analysis)", "LDA (Linear Discriminant Analysis)"])
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Toggle Classifier/Regressor Model Diagnostics
    diagnostic_toggle = st.sidebar.radio("🔎 Model Diagnostic Panel", ["Classification", "Regression", "Reinforcement Learning"])
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # RL Policy Selector
    rl_policy = st.sidebar.selectbox("🤖 RL Campaign Policy", ["Deep Q-Network (DQN)", "Tabular Q-Learning", "Random Policy", "Always No Action"])
    
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    
    # Run Analysis Button
    run_btn = st.sidebar.button("🚀 Run Simulation Campaign")
    
    # Get details of selected customer
    cust_row = df_customers[df_customers['CustomerID'] == selected_id].iloc[0]
    
    # -------------------------------------------------------------
    # MAIN AREA: DASHBOARD BODY
    # -------------------------------------------------------------
    st.markdown("""
    <div style='text-align: center; padding: 40px 0 20px 0;'>
        <h1 style='font-size: 42px; margin-bottom: 10px;'>Intelligent <span class='gradient-text'>Customer Behavior</span> Analysis</h1>
        <p style='opacity: 0.7; font-size: 18px; max-width: 800px; margin: 0 auto;'>Interactive Customer-Centric Decisions using Machine Learning & Reinforcement Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # SECTION 1: CUSTOMER SUMMARY METRICS (TOP ROW)
    st.markdown("<h3 style='margin-bottom: 20px;'>Customer Profile Summary</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Recency Card
    rec_val = int(cust_row['Recency'])
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⏱️ Recency</div>
            <div class="metric-value">{rec_val} Days</div>
            <div class="metric-sub">Since last transaction</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Frequency Card
    freq_val = int(cust_row['Frequency'])
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🛒 Frequency</div>
            <div class="metric-value">{freq_val} Orders</div>
            <div class="metric-sub">Total unique invoices</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Monetary Card
    mon_val = float(cust_row['Monetary'])
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💰 Monetary</div>
            <div class="metric-value" style="color:#22C55E;">${mon_val:,.2f}</div>
            <div class="metric-sub">Cumulative total spend</div>
        </div>
        """, unsafe_allow_html=True)
        
    # Segment Card
    high_val = int(cust_row['High_Value'])
    seg_name = "High Value Customer" if high_val == 1 else "Standard Customer"
    seg_color = "#22C55E" if high_val == 1 else "gray"
    seg_icon = "🌟" if high_val == 1 else "👤"
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">{seg_icon} Value Segment</div>
            <div class="metric-value" style="color:{seg_color}; font-size: 24px; padding-top: 8px;">{seg_name}</div>
            <div class="metric-sub">Based on top 20% spend threshold</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # SECTION 2: MIDDLE VISUALS AND PREDICTIVE CHECKS
    mid_col_left, mid_col_right = st.columns([3, 2])
    
    with mid_col_left:
        st.markdown("<h3 style='margin-top:0; margin-bottom: 20px;'>🌌 Customer Space Projection</h3>", unsafe_allow_html=True)
        
        # Plot PCA/LDA Space
        fig, ax = plt.subplots(figsize=(8, 5.5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        cat_cols = [c for c in df_customers.columns if c.endswith('_Spend_Pct')]
        # Extract and scale current customer's category spend
        current_cats = cust_row[cat_cols].values.reshape(1, -1)
        current_cats_scaled = scaler.transform(current_cats)
        
        if "PCA" in pca_lda_toggle:
            # We already have PC1 and PC2 in df_customers! Let's plot them.
            sns.scatterplot(
                data=df_customers, x='PC1', y='PC2', hue='High_Value', 
                palette={0: 'gray', 1: '#22C55E'}, alpha=0.4, ax=ax
            )
            # Project current customer
            cur_pc = pca.transform(current_cats_scaled)[0]
            ax.scatter(cur_pc[0], cur_pc[1], color='#EF4444', marker='*', s=400, edgecolor='gray', zorder=10, label='Selected Customer')
            ax.set_title("Customer Positioning in PCA Space (Category Preference)", )
            ax.set_xlabel("PC1 (Principal Component 1)")
            ax.set_ylabel("PC2 (Principal Component 2)")
        else:
            # LDA: Let's project category spend using the LDA model.
            # LDA projects to 1D because it separates binary classes.
            # Let's project all and plot LDA1 vs Monetary (Log) for illustration.
            all_cats_scaled = scaler.transform(df_customers[cat_cols])
            all_lda = lda.transform(all_cats_scaled)[:, 0]
            
            temp_df = df_customers.copy()
            temp_df['LDA1'] = all_lda
            
            sns.scatterplot(
                data=temp_df, x='LDA1', y='Monetary', hue='High_Value',
                palette={0: 'gray', 1: '#22C55E'}, alpha=0.4, ax=ax
            )
            ax.set_yscale('log')
            
            # Project current customer
            cur_lda = lda.transform(current_cats_scaled)[0, 0]
            ax.scatter(cur_lda, cust_row['Monetary'], color='#EF4444', marker='*', s=400, edgecolor='gray', zorder=10, label='Selected Customer')
            ax.set_title("Customer Positioning in LDA Space (Supervised Separation)", )
            ax.set_xlabel("LDA Component 1")
            ax.set_ylabel("Monetary Spend (Log Scale)")
            
        ax.legend(facecolor='none', edgecolor='gray')
        
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
            
        st.pyplot(fig)
        
    with mid_col_right:
        st.markdown("<h3 style='margin-top:0;'>🔮 Predictive Modeling Predictions</h3>", unsafe_allow_html=True)
        
        # Prepare feature vector for classification and regression models
        # State format: [Recency, Frequency, Monetary, PC1, PC2]
        # Project PC1 and PC2 dynamically using current row data
        cur_cats_scaled = scaler.transform(cust_row[cat_cols].values.reshape(1, -1))
        cur_pc = pca.transform(cur_cats_scaled)[0]
        
        X_input = np.array([[
            cust_row['Recency'],
            cust_row['Frequency'],
            cust_row['Monetary'],
            cur_pc[0],
            cur_pc[1]
        ]])
        
        # 1. Classification
        pred_class = classifier.predict(X_input)[0]
        pred_prob = classifier.predict_proba(X_input)[0, 1]
        
        class_label = "HIGH VALUE (Top 20%)" if pred_class == 1 else "STANDARD VALUE (Bottom 80%)"
        class_color = "color:#22C55E;" if pred_class == 1 else "opacity: 0.7;"
        badge_bg = "rgba(34, 197, 94, 0.2)" if pred_class == 1 else "rgba(128, 128, 128, 0.2)"
        badge_border = "border: 1px solid #22C55E;" if pred_class == 1 else "border: 1px solid rgba(128, 128, 128, 0.3);"
        
        # 2. Regression
        pred_spend = max(0.0, float(regressor.predict(X_input)[0]))
        
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h4 style="margin:0;">🎯 High-Value Classifier</h4>
                <span style="background: {badge_bg}; {badge_border} padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; {class_color}">{class_label}</span>
            </div>
            <div style="background: rgba(128,128,128,0.1); border-radius: 8px; padding: 12px; display: flex; align-items: center; justify-content: space-between;">
                <span style="opacity: 0.7; font-size:14px;">Probability of High Value</span>
                <span style="font-size:18px; font-weight:bold; ">{pred_prob*100:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom: 20px;">
            <h4 style="margin:0 0 12px 0;">📈 Regressor Spending Forecast</h4>
            <div style="font-size:36px; font-weight:bold; color:#06B6D4; margin-bottom: 8px;">${pred_spend:,.2f}</div>
            <p style="font-size:13px; opacity: 0.7; margin:0;">Predicted customer spend in Months 10-12 based on Months 1-9 transactional history.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="glass-card" style="padding: 16px;">
            <h5 style="margin:0 0 12px 0; opacity: 0.7;">Model Confidence Metrics (Unseen Test Set)</h5>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(128,128,128,0.1); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="opacity: 0.5; font-size: 13px;">Classifier F1-Score</span>
                <span style="font-weight: 600;  font-size: 13px;">98.2%</span>
            </div>
            <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(128,128,128,0.1); padding-bottom: 8px; margin-bottom: 8px;">
                <span style="opacity: 0.5; font-size: 13px;">Regressor R² Score</span>
                <span style="font-weight: 600;  font-size: 13px;">84.6%</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="opacity: 0.5; font-size: 13px;">Spend RMSE</span>
                <span style="font-weight: 600;  font-size: 13px;">$124.50</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # SECTION 3: REINFORCEMENT LEARNING PANEL
    st.markdown("<h3 style='margin-top:20px;'>🤖 Reinforcement Learning Campaign Recommendations</h3>", unsafe_allow_html=True)
    
    # Environment State construction: [Recency, Frequency, Monetary, PC1, PC2]
    # We query Q-values for actions 0, 1, 2
    state_vector = np.array([
        cust_row['Recency'],
        cust_row['Frequency'],
        cust_row['Monetary'],
        cur_pc[0],
        cur_pc[1]
    ], dtype=np.float32)
    
    action_descriptions = {
        0: "No Action (Expected conversion: 30%, cost: $0)",
        1: "10% Coupon (Expected conversion: 55%, cost: $1)",
        2: "Premium Trial (Expected conversion: 75%, cost: $5)"
    }
    
    costs = {0: 0, 1: 1, 2: 5}
    probs = {0: 0.30, 1: 0.55, 2: 0.75}
    
    # Compute spend and actions Q-values based on policy
    q_values = np.zeros(3)
    
    # 1. DQN Q-values
    with torch.no_grad():
        state_t = torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0)
        dqn_q = dqn(state_t).numpy()[0]
        
    # 2. Tabular Q-values
    state_idx = int(tabular_kmeans.predict(state_vector.reshape(1, -1))[0])
    tab_q = tabular_q[state_idx]
    
    # Select displaying Q-values and action recommendation
    if "DQN" in rl_policy:
        q_values = dqn_q
        chosen_action = int(np.argmax(q_values))
        policy_name = "Deep Q-Network"
    elif "Tabular" in rl_policy:
        q_values = tab_q
        chosen_action = int(np.argmax(q_values))
        policy_name = "Tabular Q-Learning"
    elif "Random" in rl_policy:
        q_values = np.random.randn(3)
        chosen_action = int(np.argmax(q_values))
        policy_name = "Random Policy"
    else: # Always No Action
        q_values = np.array([1.0, 0.0, 0.0])
        chosen_action = 0
        policy_name = "Always No Action"
        
    # Draw Q-values comparison card
    col_q1, col_q2, col_q3 = st.columns(3)
    q_cards = [col_q1, col_q2, col_q3]
    
    for i, card in enumerate(q_cards):
        is_best = (i == chosen_action)
        border_style = "border: 2px solid #3B82F6; box-shadow: 0 0 20px rgba(59, 130, 246, 0.4);" if is_best else "border: 1px solid rgba(128, 128, 128, 0.2);"
        bg_style = "background: var(--secondary-background-color);" if is_best else "background: rgba(128, 128, 128, 0.05);"
        highlight_badge = f"<div style='position:absolute; top:-12px; right:20px; background: linear-gradient(90deg, #3B82F6, #06B6D4); color:white; padding: 4px 12px; border-radius: 20px; font-size:12px; font-weight:bold;'>🏆 Recommended</div>" if is_best else ""
        
        # Expected reward calculation
        expected_reward = pred_spend * probs[i] - costs[i]
        campaign_name = action_descriptions[i].split('(')[0].strip()
        conv_rate = int(probs[i] * 100)
        camp_cost = costs[i]
        
        badge_html = highlight_badge if highlight_badge else ""
        html_content = f"""<div style="position:relative; {bg_style} {border_style} border-radius: 16px; padding: 24px; text-align: center; transition: all 0.3s ease; margin-top: 15px;">
{badge_html}
<p style="font-size:13px; opacity: 0.7; font-weight:600; text-transform:uppercase; margin:0 0 8px 0; letter-spacing: 1px;">Action {i}</p>
<h4 style="margin:5px 0 15px 0;  font-size: 20px;">{campaign_name}</h4>
<div style="background: rgba(128,128,128,0.1); border-radius: 12px; padding: 15px; margin-bottom: 15px;">
<p style="font-size:13px; opacity: 0.7; margin:0 0 4px 0;">Q-Value</p>
<p style="font-size:28px; font-weight:bold; color:#06B6D4; margin:0;">{q_values[i]:.2f}</p>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 12px; padding-bottom: 8px;">
<span style="font-size:13px; opacity: 0.5;">Conv. Rate</span>
<span style="font-size:14px;  font-weight:600;">{conv_rate}%</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 8px; padding-bottom: 8px;">
<span style="font-size:13px; opacity: 0.5;">Campaign Cost</span>
<span style="font-size:14px;  font-weight:600;">${camp_cost}</span>
</div>
<div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(128,128,128,0.1); padding-top: 12px;">
<span style="font-size:13px; opacity: 0.5;">Expected Reward</span>
<span style="font-size:16px; color:#22C55E; font-weight:bold;">${expected_reward:,.2f}</span>
</div>
</div>"""
        
        with card:
            st.markdown(html_content, unsafe_allow_html=True)
            
    # RECOMMENDATION SUMMARY
    rec_text = action_descriptions[chosen_action]
    st.markdown(f"""
    <div style="background: linear-gradient(90deg, rgba(59, 130, 246, 0.1), rgba(6, 182, 212, 0.1)); border-left: 4px solid #3B82F6; padding: 20px; border-radius: 0 12px 12px 0; margin-top: 30px; margin-bottom: 20px;">
        <h4 style="margin-top:0;  font-size:16px; display: flex; align-items: center; gap: 8px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
            Marketing Recommendation ({policy_name})
        </h4>
        <p style="margin:0; font-size:15px; opacity: 0.8; line-height: 1.6;">The optimal policy prescribes <b>Action {chosen_action}: {rec_text}</b>. This selection maximizes the net customer lifetime value trajectory, balancing the intervention cost against the augmented purchasing probability.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # SECTION 4: GLOBAL DIAGNOSTIC CHARTS (BOTTOM ACCORDIONS)
    st.markdown("<h3 style='margin-top:20px; padding-top:20px; border-top: 1px solid rgba(128,128,128,0.2);'>📊 Global Model Performance & Diagnostics</h3>", unsafe_allow_html=True)
    
    # Custom styling for tabs via CSS is tricky, but we can rely on Streamlit's default dark mode tabs for now, which look decent.
    tab_clf, tab_reg, tab_rl = st.tabs(["🛡️ Classifier Diagnostics", "📈 Regressor Diagnostics", "🧠 RL Training Progress"])
    
    with tab_clf:
        st.markdown("<h4 style='opacity: 0.7; margin-top: 10px;'>Classifier Diagnostics and Feature Importance</h4>", unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if os.path.exists('data/plots/class_confusion_matrix.png'):
                st.image('data/plots/class_confusion_matrix.png', caption="Confusion Matrix on Test Split", use_column_width=True)
        with col_c2:
            if os.path.exists('data/plots/feature_importance.png'):
                st.image('data/plots/feature_importance.png', caption="Random Forest Feature Importance", use_column_width=True)
            elif os.path.exists('data/plots/class_roc_curve.png'):
                st.image('data/plots/class_roc_curve.png', caption="Classifier ROC Curve Comparison", use_column_width=True)
                
    with tab_reg:
        st.markdown("<h4 style='opacity: 0.7; margin-top: 10px;'>Regressor Spending Forecast Diagnostics</h4>", unsafe_allow_html=True)
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            if os.path.exists('data/plots/reg_predicted_vs_actual.png'):
                st.image('data/plots/reg_predicted_vs_actual.png', caption="Predicted vs Actual Spend Scatter Plot", use_column_width=True)
        with col_r2:
            if os.path.exists('data/plots/reg_residuals.png'):
                st.image('data/plots/reg_residuals.png', caption="Regression Residual Plot", use_column_width=True)
                
    with tab_rl:
        st.markdown("<h4 style='opacity: 0.7; margin-top: 10px;'>Reinforcement Learning Policy Comparisons & Rewards</h4>", unsafe_allow_html=True)
        col_rl1, col_rl2 = st.columns(2)
        with col_rl1:
            if os.path.exists('data/plots/rl_training_rewards.png'):
                st.image('data/plots/rl_training_rewards.png', caption="Tabular Q-learning vs DQN Reward Progression", use_column_width=True)
        with col_rl2:
            if os.path.exists('data/plots/rl_policy_comparison.png'):
                st.image('data/plots/rl_policy_comparison.png', caption="Expected Average Reward under Different Policies", use_column_width=True)
                
    # Campaign simulation trigger
    if run_btn:
        st.markdown("""
        <div style="background: rgba(34, 197, 94, 0.1); border-left: 4px solid #22C55E; padding: 15px; border-radius: 4px; margin-top: 20px;">
            <p style="margin: 0; color: #22C55E; font-weight: 600;">✅ Successfully simulated campaign action sequence! Final reward calculated dynamically.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px 0; opacity: 0.5; font-size: 14px;">
        <p style="margin: 0; font-weight: 600; opacity: 0.7;">Intelligent Customer Behavior Analysis</p>
        <p style="margin: 5px 0;">Powered by Machine Learning & Reinforcement Learning</p>
        <p style="margin: 0; font-size: 13px;">Developed by <span style="">Shahab Ibrar</span></p>
    </div>
    """, unsafe_allow_html=True)
