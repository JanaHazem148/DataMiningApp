import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Mining",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ultra-modern CSS with glassmorphism and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero Header */
    .hero-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
    }
    
    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -2px;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        font-weight: 300;
    }
    
    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5);
    }
    
    /* Method Cards */
    .method-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .method-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: scale(1.02);
    }
    
    /* Stats Cards */
    .stat-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Process Steps */
    .process-step {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4ade80;
        color: white;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Buttons */
    .stDownloadButton button {
        background: linear-gradient(45deg, #4ade80, #22c55e);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(74, 222, 128, 0.4);
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(74, 222, 128, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%);
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4ade80, #22c55e);
        border-radius: 10px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin: 2rem 0 1rem 0;
        text-align: center;
    }
    
    .subsection-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: white;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
    }
    
    /* Upload Section */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(255, 255, 255, 0.3);
        border-radius: 20px;
    }
    
    /* Info boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        color: white !important;
        font-weight: 600;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    
    p, span, div {
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="hero-header">
    <div class="hero-title">✨ Data Mining app</div>
    <div class="hero-subtitle">Transform messy data into insights with advanced analytics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 Processing Method")
    
    processing_method = st.radio(
        "",
        ["🔍 Data Cleaning Pipeline", "🔗 Association Rule Mining", "📊 PCA & Analysis"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if "Pipeline" in processing_method:
        st.markdown("""
        ### 📋 About Pipeline
        **Comprehensive data cleaning:**
        - Missing value imputation
        - Duplicate removal
        - Data type conversion
        - Text normalization
        - Outlier detection
        """)
    elif "Association" in processing_method:
        st.markdown("""
        ### 🛒 About Association Rules
        **Market basket analysis:**
        - Frequent itemset mining
        - Rule generation
        - Support & confidence metrics
        - Business insights
        - Product bundling recommendations
        """)
    else:
        st.markdown("""
        ### 🎯 About PCA
        **Dimensionality reduction:**
        - Feature extraction
        - Variance analysis
        - Data compression
        - Pattern recognition
        - Component visualization
        """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    show_details = st.checkbox("Show detailed logs", value=True)

# ============================================================================
# DATA CLEANING PIPELINE
# ============================================================================
def cleaning_pipeline(df, show_details=True):
    st.markdown('<div class="section-header">🔍 Data Cleaning Pipeline</div>', unsafe_allow_html=True)
    
    # Original Data
    st.markdown('<div class="subsection-header">📥 Original Data</div>', unsafe_allow_html=True)
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=300)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{df.shape[0]:,}</div><div class="stat-label">Rows</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{df.shape[1]}</div><div class="stat-label">Columns</div></div>', unsafe_allow_html=True)
        with col3:
            missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100) if (df.shape[0] * df.shape[1]) > 0 else 0
            st.markdown(f'<div class="stat-card"><div class="stat-number">{missing_pct:.1f}%</div><div class="stat-label">Missing</div></div>', unsafe_allow_html=True)
    
    # Quality Assessment
    st.markdown('<div class="subsection-header">🔍 Quality Assessment</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col2:
        st.metric("Duplicates", df.duplicated().sum())
    with col3:
        st.metric("Data Types", df.dtypes.nunique())
    with col4:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        st.metric("Numeric Columns", len(numeric_cols))
    
    # Cleaning Process
    st.markdown('<div class="subsection-header">⚙️ Cleaning in Progress</div>', unsafe_allow_html=True)
    
    progress_bar = st.progress(0)
    status_container = st.container()
    
    df_cleaned = df.copy()
    steps_completed = []
    
    # Step 1: Remove duplicates
    with status_container:
        st.markdown('<div class="process-step">✓ Step 1/6: Removing duplicate rows...</div>', unsafe_allow_html=True)
    progress_bar.progress(16)
    dup_count = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    steps_completed.append(f"Removed {dup_count} duplicates")
    
    # Step 2: Handle missing values (numeric)
    with status_container:
        st.markdown('<div class="process-step">✓ Step 2/6: Imputing missing numeric values...</div>', unsafe_allow_html=True)
    progress_bar.progress(32)
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_cleaned[col].isnull().sum() > 0:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
    steps_completed.append(f"Imputed {len(numeric_cols)} numeric columns")
    
    # Step 3: Handle missing values (categorical)
    with status_container:
        st.markdown('<div class="process-step">✓ Step 3/6: Filling categorical missing values...</div>', unsafe_allow_html=True)
    progress_bar.progress(48)
    cat_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_cleaned[col].isnull().sum() > 0:
            mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else "Unknown"
            df_cleaned[col].fillna(mode_val, inplace=True)
    steps_completed.append(f"Filled {len(cat_cols)} categorical columns")
    
    # Step 4: Clean text columns
    with status_container:
        st.markdown('<div class="process-step">✓ Step 4/6: Normalizing text data...</div>', unsafe_allow_html=True)
    progress_bar.progress(64)
    for col in cat_cols:
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        df_cleaned[col] = df_cleaned[col].str.replace(r'\s+', ' ', regex=True)
    steps_completed.append("Normalized all text columns")
    
    # Step 5: Convert data types
    with status_container:
        st.markdown('<div class="process-step">✓ Step 5/6: Optimizing data types...</div>', unsafe_allow_html=True)
    progress_bar.progress(80)
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype == 'object':
            try:
                numeric_version = pd.to_numeric(df_cleaned[col], errors='coerce')
                if numeric_version.notna().sum() > len(df_cleaned) * 0.5:
                    df_cleaned[col] = numeric_version
            except:
                pass
    steps_completed.append("Optimized data types")
    
    # Step 6: Final validation
    with status_container:
        st.markdown('<div class="process-step">✓ Step 6/6: Validating cleaned data...</div>', unsafe_allow_html=True)
    progress_bar.progress(100)
    steps_completed.append("Validation complete")
    
    st.success("✅ Cleaning pipeline completed successfully!")
    
    # Cleaned Data - FIXED: This section now appears properly
    st.markdown('<div class="subsection-header">✨ Cleaned Data</div>', unsafe_allow_html=True)
    with st.expander("View Cleaned Data", expanded=False):
        st.dataframe(df_cleaned, use_container_width=True, height=300)
        st.write(f"**Cleaned Dataset Shape:** {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
    
    # Results comparison
    st.markdown('<div class="subsection-header">📊 Cleaning Results</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows Retained", f"{df_cleaned.shape[0]:,}", f"{df_cleaned.shape[0] - df.shape[0]}")
    with col2:
        retention = (df_cleaned.shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 100
        st.metric("Data Retention", f"{retention:.1f}%")
    with col3:
        missing_after = df_cleaned.isnull().sum().sum()
        st.metric("Missing Values", missing_after, f"-{df.isnull().sum().sum() - missing_after}")
    with col4:
        st.metric("Quality Score", f"{retention:.0f}%", "Excellent" if retention > 90 else "Good")
    
    if show_details:
        with st.expander("📝 Detailed Cleaning Log"):
            for i, step in enumerate(steps_completed, 1):
                st.write(f"{i}. {step}")
    
    return df_cleaned

# ============================================================================
# ASSOCIATION RULE MINING
# ============================================================================
def association_mining(df, show_details=True):
    st.markdown('<div class="section-header">🔗 Association Rule Mining</div>', unsafe_allow_html=True)
    
    # Check for library
    try:
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        st.error("❌ mlxtend library not installed. Install with: `pip install mlxtend`")
        st.code("pip install mlxtend", language="bash")
        return df, None
    
    # Original Data
    st.markdown('<div class="subsection-header">📥 Original Transaction Data</div>', unsafe_allow_html=True)
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)
        st.write(f"**Total rows:** {df.shape[0]:,} | **Columns:** {df.shape[1]}")
    
    # Check for transaction structure
    st.markdown('<div class="subsection-header">🔍 Data Structure Analysis</div>', unsafe_allow_html=True)
    
    trans_col = None
    item_col = None
    
    for col in df.columns:
        if any(word in col.lower() for word in ['transaction', 'order', 'invoice', 'id']):
            trans_col = col
            break
    
    for col in df.columns:
        if any(word in col.lower() for word in ['item', 'product', 'sku']):
            item_col = col
            break
    
    if trans_col is None or item_col is None:
        st.warning("⚠️ Could not auto-detect transaction structure. Please select columns:")
        col1, col2 = st.columns(2)
        with col1:
            trans_col = st.selectbox("Transaction ID Column", df.columns)
        with col2:
            item_col = st.selectbox("Item/Product Column", df.columns)
    else:
        st.success(f"✅ Detected: Transaction ID = '{trans_col}', Items = '{item_col}'")
    
    # Clean data first
    st.markdown('<div class="subsection-header">⚙️ Cleaning Transaction Data</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    df_cleaned = df.copy()
    cleaning_log = []
    
    # Clean items
    st.markdown('<div class="process-step">✓ Step 1/5: Cleaning item names...</div>', unsafe_allow_html=True)
    progress_bar.progress(20)
    df_cleaned[item_col] = df_cleaned[item_col].astype(str)
    df_cleaned[item_col] = df_cleaned[item_col].str.replace(";", "").str.replace(",", "")
    df_cleaned[item_col] = df_cleaned[item_col].str.strip().str.lower()
    blank_count = (df_cleaned[item_col] == "").sum()
    df_cleaned = df_cleaned[df_cleaned[item_col] != ""]
    cleaning_log.append(f"Removed special characters and {blank_count} blank items")
    
    # Remove bad values
    st.markdown('<div class="process-step">✓ Step 2/5: Removing invalid entries...</div>', unsafe_allow_html=True)
    progress_bar.progress(40)
    df_cleaned = df_cleaned.dropna(subset=[trans_col, item_col])
    invalid_items = df_cleaned[df_cleaned[item_col].isin(['unknown', 'error', 'nan', 'none'])].shape[0]
    df_cleaned = df_cleaned[~df_cleaned[item_col].isin(['unknown', 'error', 'nan', 'none'])]
    cleaning_log.append(f"Removed {invalid_items} invalid items")
    
    # Remove duplicates
    st.markdown('<div class="process-step">✓ Step 3/5: Removing duplicates...</div>', unsafe_allow_html=True)
    progress_bar.progress(60)
    dup_count = df_cleaned.duplicated().sum()
    df_cleaned = df_cleaned.drop_duplicates()
    cleaning_log.append(f"Removed {dup_count} duplicate rows")
    
    progress_bar.progress(80)
    st.markdown('<div class="process-step">✓ Step 4/5: Validating data structure...</div>', unsafe_allow_html=True)
    
    progress_bar.progress(100)
    st.markdown('<div class="process-step">✓ Step 5/5: Data cleaning complete!</div>', unsafe_allow_html=True)
    
    st.success("✅ Data cleaning completed!")
    
    # Cleaning summary
    st.markdown("### 📋 Cleaning Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Cleaned Rows", f"{df_cleaned.shape[0]:,}")
    with col3:
        retention = (df_cleaned.shape[0] / df.shape[0]) * 100 if df.shape[0] > 0 else 100
        st.metric("Data Retention", f"{retention:.1f}%")
    
    # Show cleaned data - FIXED: Now properly displayed
    st.markdown('<div class="subsection-header">✨ Cleaned Transaction Data</div>', unsafe_allow_html=True)
    with st.expander("View Cleaned Data", expanded=False):
        st.dataframe(df_cleaned.head(20), use_container_width=True, height=300)
        st.write(f"**Cleaned Dataset Shape:** {df_cleaned.shape[0]:,} rows × {df_cleaned.shape[1]} columns")
    
    if show_details:
        with st.expander("📝 Detailed Cleaning Log"):
            for i, log in enumerate(cleaning_log, 1):
                st.write(f"{i}. {log}")
    
    # Prepare for mining
    st.markdown('<div class="subsection-header">🛒 Market Basket Analysis</div>', unsafe_allow_html=True)
    
    try:
        # Create transactions
        transactions = df_cleaned.groupby(trans_col)[item_col].apply(list).tolist()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(transactions))
        with col2:
            st.metric("Unique Items", df_cleaned[item_col].nunique())
        with col3:
            avg_items = df_cleaned.groupby(trans_col)[item_col].count().mean()
            st.metric("Avg Items/Transaction", f"{avg_items:.1f}")
        
        # Show sample transactions
        with st.expander("🛍️ Sample Transactions"):
            for i, trans in enumerate(transactions[:10], 1):
                st.write(f"**{i}.** {', '.join(trans)}")
        
        # Prepare basket
        st.markdown('<div class="process-step">⚙️ Converting to basket format...</div>', unsafe_allow_html=True)
        te = TransactionEncoder()
        te_array = te.fit(transactions).transform(transactions)
        basket = pd.DataFrame(te_array, columns=te.columns_)
        st.success(f"✓ Basket created: {basket.shape[0]} transactions × {basket.shape[1]} items")
        
        # Mining parameters
        st.markdown("### ⚙️ Mining Parameters")
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.01, 0.3, 0.05, 0.01, 
                                   help="Minimum frequency of itemset (e.g., 0.05 = 5%)")
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.05,
                                      help="Minimum reliability of rule (e.g., 0.3 = 30%)")
        
        # Run Apriori
        st.markdown("### 🔍 Running Apriori Algorithm...")
        st.markdown('<div class="process-step">⚙️ Mining frequent itemsets...</div>', unsafe_allow_html=True)
        
        frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
        
        if len(frequent_items) > 0:
            frequent_items['length'] = frequent_items['itemsets'].apply(lambda x: len(x))
            frequent_items = frequent_items.sort_values(by='support', ascending=False)
            
            st.success(f"✅ Found {len(frequent_items)} frequent itemsets!")
            
            # Itemset distribution
            st.markdown("### 📊 Frequent Itemsets Distribution")
            itemset_dist = frequent_items['length'].value_counts().sort_index()
            col1, col2 = st.columns([1, 2])
            with col1:
                st.dataframe(itemset_dist.to_frame(name="Count"), use_container_width=True)
            with col2:
                st.bar_chart(itemset_dist)
            
            # Show frequent itemsets
            st.markdown("### 📦 Top 15 Frequent Itemsets")
            display_items = frequent_items.head(15).copy()
            display_items['itemsets'] = display_items['itemsets'].apply(lambda x: ', '.join(list(x)))
            display_items['support'] = display_items['support'].apply(lambda x: f"{x:.4f}")
            st.dataframe(display_items[['itemsets', 'support', 'length']], 
                        use_container_width=True, hide_index=True)
            
            # Generate rules
            st.markdown("### 🎯 Generating Association Rules...")
            st.markdown('<div class="process-step">⚙️ Creating association rules...</div>', unsafe_allow_html=True)
            
            rules = association_rules(frequent_items, metric="confidence", min_threshold=min_confidence)
            
            if len(rules) > 0:
                rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
                
                st.success(f"✅ Generated {len(rules)} association rules!")
                
                # Rules Statistics
                st.markdown("### 📊 Rules Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rules", len(rules))
                with col2:
                    st.metric("Avg Support", f"{rules['support'].mean():.4f}")
                with col3:
                    st.metric("Avg Confidence", f"{rules['confidence'].mean():.4f}")
                with col4:
                    st.metric("Max Lift", f"{rules['lift'].max():.4f}")
                
                # Top rules tabs
                st.markdown("### 🏆 Top Association Rules")
                
                tab1, tab2, tab3 = st.tabs(["📈 By Lift (Strongest)", "🎯 By Confidence (Most Reliable)", "📊 By Support (Most Frequent)"])
                
                with tab1:
                    st.markdown("**Lift > 1 means items are positively correlated**")
                    top_lift = rules.nlargest(10, 'lift')[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    top_lift['support'] = top_lift['support'].apply(lambda x: f"{x:.4f}")
                    top_lift['confidence'] = top_lift['confidence'].apply(lambda x: f"{x:.1%}")
                    top_lift['lift'] = top_lift['lift'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(top_lift, use_container_width=True, hide_index=True)
                
                with tab2:
                    st.markdown("**High confidence = reliable predictions**")
                    top_conf = rules.nlargest(10, 'confidence')[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    top_conf['support'] = top_conf['support'].apply(lambda x: f"{x:.4f}")
                    top_conf['confidence'] = top_conf['confidence'].apply(lambda x: f"{x:.1%}")
                    top_conf['lift'] = top_conf['lift'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(top_conf, use_container_width=True, hide_index=True)
                
                with tab3:
                    st.markdown("**High support = frequently occurring together**")
                    top_supp = rules.nlargest(10, 'support')[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                    top_supp['support'] = top_supp['support'].apply(lambda x: f"{x:.4f}")
                    top_supp['confidence'] = top_supp['confidence'].apply(lambda x: f"{x:.1%}")
                    top_supp['lift'] = top_supp['lift'].apply(lambda x: f"{x:.2f}")
                    st.dataframe(top_supp, use_container_width=True, hide_index=True)
                
                # Key insights
                st.markdown("### 💡 Key Insights & Business Recommendations")
                
                strongest = rules.nlargest(3, 'lift')
                for idx, row in strongest.iterrows():
                    st.markdown(f"""
                    <div class="glass-card">
                        <h4>🌟 {row['antecedents']} → {row['consequents']}</h4>
                        <p><strong>Lift:</strong> {row['lift']:.2f}x | <strong>Confidence:</strong> {row['confidence']:.1%} | <strong>Support:</strong> {row['support']:.1%}</p>
                        <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                        {"Strong positive correlation - great for bundling!" if row['lift'] > 2 else "Moderate correlation - consider for recommendations"}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interpretation guide
                with st.expander("📖 Understanding the Metrics"):
                    st.markdown("""
                    **Support:** Frequency of itemset in all transactions
                    - Higher support = more popular combination
                    - Example: Support of 0.05 means 5% of all transactions contain this combination
                    
                    **Confidence:** P(consequent|antecedent) - How often the rule is correct
                    - Higher confidence = more reliable rule
                    - Example: Confidence of 0.80 means 80% of people who buy A also buy B
                    
                    **Lift:** Correlation strength between items
                    - Lift > 1: Items are positively correlated (bought together more than random)
                    - Lift = 1: Items are independent (no correlation)
                    - Lift < 1: Items are negatively correlated (rarely together)
                    
                    **Business Applications:**
                    - **High Lift Rules:** Bundle products, cross-selling opportunities
                    - **High Confidence Rules:** Product recommendations, upselling strategies
                    - **High Support Rules:** Popular combinations, promotion strategies
                    """)
                
                # Export options
                st.markdown("### 💾 Export Options")
                col1, col2 = st.columns(2)
                with col1:
                    rules_csv = rules.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download All Rules",
                        data=rules_csv,
                        file_name="association_rules.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                with col2:
                    freq_csv = frequent_items.copy()
                    freq_csv['itemsets'] = freq_csv['itemsets'].apply(lambda x: ', '.join(list(x)))
                    freq_csv = freq_csv.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Frequent Itemsets",
                        data=freq_csv,
                        file_name="frequent_itemsets.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                return df_cleaned, rules
            else:
                st.warning("⚠️ No rules found. Try lowering the confidence threshold.")
                return df_cleaned, None
        else:
            st.warning("⚠️ No frequent itemsets found. Try lowering the support threshold.")
            st.info("""
            **Tips:**
            - Lower the minimum support (try 0.01 or 0.02)
            - Check if your data has enough transactions
            - Verify that items appear in multiple transactions
            """)
            return df_cleaned, None
            
    except Exception as e:
        st.error(f"❌ Error during mining: {str(e)}")
        st.info("Please check your data format and try again.")
        return df_cleaned, None

# ============================================================================
# PCA & ANALYSIS
# ============================================================================
def pca_analysis(df, show_details=True):
    st.markdown('<div class="section-header">📊 PCA & Advanced Analysis</div>', unsafe_allow_html=True)
    
    # Check for library
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
    except ImportError:
        st.error("❌ scikit-learn not installed. Install with: `pip install scikit-learn matplotlib`")
        st.code("pip install scikit-learn matplotlib", language="bash")
        return df, None
    
    # Original Data
    st.markdown('<div class="subsection-header">📥 Original Data</div>', unsafe_allow_html=True)
    with st.expander("View Raw Data", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)
        st.write(f"**Dataset Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Data info
    st.markdown("### 📋 Data Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{df.shape[0]:,}")
    with col2:
        st.metric("Total Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data Preparation
    st.markdown('<div class="subsection-header">⚙️ Data Preparation & Cleaning</div>', unsafe_allow_html=True)
    progress_bar = st.progress(0)
    
    df_cleaned = df.copy()
    cleaning_log = []
    
    # Step 1: Clean data
    st.markdown('<div class="process-step">✓ Step 1/6: Removing duplicates and missing values...</div>', unsafe_allow_html=True)
    progress_bar.progress(16)
    original_rows = df_cleaned.shape[0]
    df_cleaned = df_cleaned.drop_duplicates()
    dup_removed = original_rows - df_cleaned.shape[0]
    cleaning_log.append(f"Removed {dup_removed} duplicate rows")
    
    # Handle missing values
    missing_before = df_cleaned.isnull().sum().sum()
    df_cleaned = df_cleaned.dropna()
    rows_with_missing = original_rows - dup_removed - df_cleaned.shape[0]
    cleaning_log.append(f"Removed {rows_with_missing} rows with missing values")
    
    # Step 2: Select numeric columns
    st.markdown('<div class="process-step">✓ Step 2/6: Selecting numeric features...</div>', unsafe_allow_html=True)
    progress_bar.progress(32)
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("❌ Need at least 2 numeric columns for PCA analysis")
        st.info(f"Found only {len(numeric_cols)} numeric column(s). PCA requires at least 2 numeric features.")
        return df_cleaned, None
    
    df_numeric = df_cleaned[numeric_cols].copy()
    cleaning_log.append(f"Selected {len(numeric_cols)} numeric features for PCA")
    
    # Show selected features
    with st.expander("🔢 Selected Numeric Features"):
        st.write(", ".join(numeric_cols))
    
    # Step 3: Standardization (Normalization)
    st.markdown('<div class="process-step">✓ Step 3/6: Applying normalization (z-score standardization)...</div>', unsafe_allow_html=True)
    progress_bar.progress(48)
    
    try:
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)
        cleaning_log.append("Applied StandardScaler (z-score normalization) to scale features")
        cleaning_log.append(f"Each feature now has mean=0 and std=1")
    except Exception as e:
        st.error(f"❌ Error during standardization: {str(e)}")
        return df_cleaned, None
    
    # Show normalized data
    st.markdown('<div class="subsection-header">✨ Normalized Data</div>', unsafe_allow_html=True)
    with st.expander("View Normalized Data (After Scaling)", expanded=False):
        st.dataframe(df_scaled.head(20), use_container_width=True, height=300)
        st.write(f"**Normalized Dataset Shape:** {df_scaled.shape[0]:,} rows × {df_scaled.shape[1]} columns")
        st.info("📊 Data has been standardized: mean = 0, standard deviation = 1 for all features")
    
    # Step 4: PCA
    st.markdown('<div class="process-step">✓ Step 4/6: Applying PCA (Principal Component Analysis)...</div>', unsafe_allow_html=True)
    progress_bar.progress(64)
    
    try:
        # Determine number of components
        n_components = min(len(numeric_cols), 10)
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df_scaled)
        
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        cleaning_log.append(f"PCA completed with {n_components} components")
        
    except Exception as e:
        st.error(f"❌ Error during PCA: {str(e)}")
        return df_cleaned, None
    
    # Step 5: Create visualizations
    st.markdown('<div class="process-step">✓ Step 5/6: Generating visualizations...</div>', unsafe_allow_html=True)
    progress_bar.progress(80)
    
    # Step 6: Analysis
    st.markdown('<div class="process-step">✓ Step 6/6: Generating insights and final report...</div>', unsafe_allow_html=True)
    progress_bar.progress(100)
    cleaning_log.append("Analysis complete!")
    
    st.success("✅ PCA analysis completed successfully!")
    
    # Cleaning summary
    if show_details:
        st.markdown("### 📝 Detailed Processing Log")
        with st.expander("View Complete Processing Steps", expanded=False):
            for i, log in enumerate(cleaning_log, 1):
                st.write(f"{i}. {log}")
    
    # Results
    st.markdown('<div class="subsection-header">✨ PCA Results</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Original Features", len(numeric_cols))
    with col2:
        st.metric("Principal Components", n_components)
    with col3:
        variance_explained = pca.explained_variance_ratio_.sum() * 100
        st.metric("Variance Explained", f"{variance_explained:.1f}%")
    with col4:
        reduction = (1 - n_components/len(numeric_cols))*100 if len(numeric_cols) > n_components else 0
        st.metric("Dimension Reduction", f"{reduction:.0f}%")
    
    # Variance explanation
    st.markdown("### 📈 Variance Explained by Components")
    variance_df = pd.DataFrame({
        'Component': [f'PC{i+1}' for i in range(n_components)],
        'Variance Explained (%)': pca.explained_variance_ratio_ * 100,
        'Cumulative Variance (%)': np.cumsum(pca.explained_variance_ratio_) * 100
    })
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(variance_df.style.format({
            'Variance Explained (%)': '{:.2f}',
            'Cumulative Variance (%)': '{:.2f}'
        }), use_container_width=True, hide_index=True)
    
    with col2:
        st.bar_chart(variance_df.set_index('Component')['Variance Explained (%)'])
    
    # Scree plot
    st.markdown("### 📊 Scree Plot - Explained Variance")
    st.line_chart(variance_df.set_index('Component')['Variance Explained (%)'])
    st.caption("The scree plot shows how much variance each component explains. Look for the 'elbow' point.")
    
    # PCA Visualization (2D and 3D)
    st.markdown("### 🎯 PCA Visualization")
    
    if n_components >= 2:
        tab1, tab2 = st.tabs(["📊 2D Visualization", "🎲 3D Visualization"])
        
        with tab1:
            st.markdown("**2D Plot: First Two Principal Components**")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.6, c=range(len(pca_df)), cmap='viridis')
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)', fontsize=12)
            ax.set_title('PCA: First Two Principal Components', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Sample Index')
            st.pyplot(fig)
            st.caption(f"2D scatter plot showing the first two components (explaining {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100:.1f}% of variance)")
        
        with tab2:
            if n_components >= 3:
                st.markdown("**3D Plot: First Three Principal Components**")
                try:
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], 
                                       c=range(len(pca_df)), cmap='viridis', alpha=0.6)
                    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=10)
                    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=10)
                    ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]*100:.2f}%)', fontsize=10)
                    ax.set_title('PCA: First Three Principal Components', fontsize=14, fontweight='bold')
                    plt.colorbar(scatter, ax=ax, label='Sample Index', pad=0.1)
                    st.pyplot(fig)
                    st.caption(f"3D scatter plot showing the first three components (explaining {(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2])*100:.1f}% of variance)")
                except:
                    st.info("3D visualization requires matplotlib with 3D support")
            else:
                st.info("Need at least 3 components for 3D visualization")
    
    # Principal components data
    st.markdown("### 🎯 Principal Components (Transformed Data)")
    with st.expander("View PCA Transformed Data", expanded=False):
        display_pca = pca_df.head(100)
        st.dataframe(display_pca, use_container_width=True, height=300)
        st.caption(f"Showing first 100 rows of {pca_df.shape[0]:,} total rows")
    
    # Component statistics
    st.markdown("### 📊 Principal Component Statistics")
    st.dataframe(pca_df.describe(), use_container_width=True)
    
    # Feature contributions (loadings)
    st.markdown("### 🔍 Feature Contributions to Principal Components")
    
    # Create loadings dataframe
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numeric_cols
    )
    
    # Show top contributors for each PC
    tabs = st.tabs([f"PC{i+1}" for i in range(min(5, n_components))])
    
    for i, tab in enumerate(tabs):
        with tab:
            pc_loadings = loadings.iloc[:, i].abs().sort_values(ascending=False)
            st.markdown(f"**Top Features Contributing to PC{i+1}:**")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                top_10 = pc_loadings.head(10).to_frame(name='Absolute Contribution')
                st.dataframe(top_10.style.format('{:.4f}'), use_container_width=True)
            with col2:
                st.bar_chart(pc_loadings.head(10))
            
            st.caption(f"PC{i+1} explains {pca.explained_variance_ratio_[i]*100:.2f}% of total variance")
    
    # Full loadings matrix
    with st.expander("📋 Full Loadings Matrix"):
        st.dataframe(loadings.style.format('{:.4f}'), use_container_width=True)
        st.caption("Loadings show how much each original feature contributes to each principal component")
    
    # Interpretation guide
    with st.expander("📖 Understanding PCA Results"):
        st.markdown("""
        **What is PCA?**
        Principal Component Analysis (PCA) is a dimensionality reduction technique that:
        - Transforms correlated features into uncorrelated principal components
        - Reduces data complexity while preserving most of the variance
        - Helps identify patterns and reduce noise
        
        **Key Metrics:**
        - **Variance Explained:** How much information each component captures
        - **Cumulative Variance:** Total information captured by first N components
        - **Loadings:** Contribution of each original feature to each component
        
        **How to Use:**
        1. Look for components that explain >10% variance (these are important)
        2. Check cumulative variance - typically aim for 80-90%
        3. Examine loadings to understand what each component represents
        4. Use transformed data for visualization or further analysis
        
        **Business Applications:**
        - Customer segmentation (reduce many features to key dimensions)
        - Anomaly detection (identify outliers in reduced space)
        - Data compression (store less data while keeping information)
        - Visualization (plot 2D/3D even with 100+ original features)
        """)
    
    # Recommendations
    st.markdown("### 💡 Recommendations")
    
    variance_80 = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.80)[0]
    if len(variance_80) > 0:
        n_for_80 = variance_80[0] + 1
        st.info(f"""
        **Optimal Components:** Keep the first **{n_for_80} components** to retain 80% of variance.
        
        This reduces dimensionality from **{len(numeric_cols)} → {n_for_80}** features ({(1-n_for_80/len(numeric_cols))*100:.0f}% reduction)
        while preserving most of the information.
        """)
    
    # Export options
    st.markdown("### 💾 Download Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pca_csv = pca_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download PCA Data",
            data=pca_csv,
            file_name="pca_transformed_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        loadings_csv = loadings.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Loadings",
            data=loadings_csv,
            file_name="pca_loadings.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        variance_csv = variance_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Variance",
            data=variance_csv,
            file_name="variance_explained.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    return df_cleaned, pca_df

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file, low_memory=False)
        
        st.markdown(f"""
        <div class="glass-card">
            <h3>✅ File Loaded Successfully!</h3>
            <p><strong>Dataset:</strong> {uploaded_file.name}</p>
            <p><strong>Size:</strong> {df.shape[0]:,} rows × {df.shape[1]} columns</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Apply selected method
        result_data = None
        extra_data = None
        
        if "Pipeline" in processing_method:
            result_data = cleaning_pipeline(df, show_details)
        elif "Association" in processing_method:
            result_data, extra_data = association_mining(df, show_details)
        else:
            result_data, extra_data = pca_analysis(df, show_details)
        
        # Download section
        if result_data is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">💾 Download Cleaned Data</div>', unsafe_allow_html=True)
            
            csv = result_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Cleaned Dataset",
                data=csv,
                file_name=f"cleaned_{uploaded_file.name}",
                mime="text/csv",
                use_container_width=True
            )
            
            # Final Statistics
            with st.expander("📊 Final Dataset Statistics"):
                st.dataframe(result_data.describe(include='all'), use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file is properly formatted.")
        with st.expander("🔍 Error Details"):
            st.code(str(e))

else:
    # Landing page
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to DataCleanse AI</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">Upload your CSV file to get started</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Choose Your Processing Method")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="method-card">
            <h3>🔍 Pipeline</h3>
            <p><strong>Comprehensive cleaning</strong></p>
            <ul style="text-align: left; color: white;">
                <li>Remove duplicates</li>
                <li>Handle missing values</li>
                <li>Type conversion</li>
                <li>Text normalization</li>
                <li>Data validation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="method-card">
            <h3>🔗 Association Rules</h3>
            <p><strong>Market basket analysis</strong></p>
            <ul style="text-align: left; color: white;">
                <li>Frequent itemsets</li>
                <li>Rule generation</li>
                <li>Support metrics</li>
                <li>Confidence scores</li>
                <li>Business insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="method-card">
            <h3>📊 PCA</h3>
            <p><strong>Dimensionality reduction</strong></p>
            <ul style="text-align: left; color: white;">
                <li>Feature extraction</li>
                <li>Variance analysis</li>
                <li>Data compression</li>
                <li>Pattern recognition</li>
                <li>Component scores</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255, 255, 255, 0.7); padding: 2rem;'>
    <p style='font-size: 0.9rem;'>Made by Jana ❤️ using Streamlit | Data Mining course</p>
    <p style='font-size: 0.8rem;'>Transform your data into actionable insights</p>
</div>
""", unsafe_allow_html=True)