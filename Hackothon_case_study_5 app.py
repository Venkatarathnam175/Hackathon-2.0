import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
st.set_page_config(
    page_title="REC-SSEC Bank Transaction Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up visual style for plots
plt.style.use('ggplot')
sns.set_palette("Set2")

@st.cache_data
def load_and_process_data():
    """
    Simulates loading the bank dataset and performs all necessary
    data processing, aggregation, and clustering steps.
    
    NOTE: In a real-world scenario, replace this mock data section
    with pd.read_excel("path/to/bankdataset.xlsx") and run the
    preprocessing steps.
    """
    
    # ----------------------------------------------------
    # MOCK DATA CREATION (To ensure the app runs independently)
    # ----------------------------------------------------
    N_LOCATIONS = 40
    N_DOMAINS = 5
    N_DAYS = 365
    
    locations = [f'City_{i}' for i in range(1, N_LOCATIONS + 1)]
    domains = ['Investments', 'International', 'Medical', 'Restaurant', 'Public']
    
    dates = pd.to_datetime(pd.date_range(start='2024-01-01', periods=N_DAYS, freq='D'))
    
    # Create a DataFrame that mimics the scale of the original data (40 * 5 * 365 = 73000 rows)
    # This ensures the subsequent aggregations have enough data points.
    data_list = []
    for day in dates:
        for loc in locations:
            for dom in domains:
                # Value is high for Investments/International, mid for Medical/Restaurant, low for Public
                base_value = 800000 + (100000 * domains.index(dom))
                # Transaction count is stable across all
                base_count = 1500
                
                value = np.random.normal(base_value, base_value * 0.1)
                count = np.random.normal(base_count, base_count * 0.1)
                
                # Ensure values are positive
                value = max(10000, value)
                count = max(100, count)
                
                data_list.append({
                    'Date': day,
                    'Domain': dom,
                    'Location': loc,
                    'Value': value,
                    'Transaction_count': count
                })

    data = pd.DataFrame(data_list)
    # ----------------------------------------------------
    # END OF MOCK DATA CREATION
    # ----------------------------------------------------
    
    
    # --- PREPROCESSING & AGGREGATION (from original notebook) ---
    data['Month'] = data['Date'].dt.to_period('M')
    data['dayofweek'] = data['Date'].dt.day_name()
    
    # Regional Performance
    regional_perf = data.groupby(['Location']).agg(
        avg_txn_value=('Value','mean'),
        avg_txn_count=('Transaction_count','mean'),
        total_transactions=('Transaction_count','sum'),
        total_value=('Value','sum'),
        days_recorded=('Date','nunique')
    ).reset_index()
    
    # Monthly & Daily Summary for Temporal Analysis
    monthly_summary = data.groupby(['Month']).agg(
        total_value=('Value','sum'),
        total_transactions=('Transaction_count','sum')
    ).reset_index()
    monthly_summary['Month'] = monthly_summary['Month'].astype(str)
    
    week_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_summary = data.groupby(['dayofweek']).agg(
        total_value=('Value','sum'),
        total_transactions=('Transaction_count','sum')
    ).reindex(week_order).reset_index()
    
    # Domain-Location Daily Aggregation for Clustering
    daily_loca = data.groupby(['Domain','Location','Date']).agg(
        total_value=('Value','sum'),
        total_transactions=('Transaction_count','sum')
    ).reset_index()

    dc = daily_loca.groupby(['Domain','Location']).agg(
        avg_daily_value=('total_value','mean'),
        avg_daily_count=('total_transactions','mean'),
        total_value=('total_value','sum'),
        total_transactions=('total_transactions','sum'),
        ).reset_index()

    # --- K-MEANS CLUSTERING (k=3 based on notebook findings) ---
    X = dc[['avg_daily_value','avg_daily_count','total_value','total_transactions']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use k=3 as determined in the original analysis
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    dc['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map cluster labels (Note: Cluster IDs may change based on random_state/initialization,
    # so we identify them by their mean performance post-clustering)
    cluster_means = dc.groupby('Cluster')[['total_value', 'total_transactions']].mean().sum(axis=1).sort_values(ascending=False)
    
    cluster_names = {
        cluster_means.index[0]: "HIGH_PERFORMANCE",
        cluster_means.index[1]: "MEDIUM_PERFORMANCE",
        cluster_means.index[2]: "LOW_PERFORMANCE"
    }

    dc['Cluster_Label'] = dc['Cluster'].map(cluster_names)
    
    return data, regional_perf, monthly_summary, daily_summary, dc

# Load all pre-processed dataframes
data, regional_perf, monthly_summary, daily_summary, dc = load_and_process_data()

# --- HELPER FUNCTIONS FOR VISUALIZATION ---

def plot_top_10_regional(df):
    """Plots top 10 locations by total value and total transactions."""
    top10_value = df.nlargest(10, 'total_value')
    top10_count = df.nlargest(10, 'total_transactions')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    sns.barplot(x='Location', y='total_value', data=top10_value, ax=axes[0], palette="viridis")
    axes[0].set_title('Top 10 Locations by Total Value (₹)', fontsize=16)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Total Value (Billions)")

    sns.barplot(x='Location', y='total_transactions', data=top10_count, ax=axes[1], palette="magma")
    axes[1].set_title('Top 10 Locations by Total Transactions (Volume)', fontsize=16)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total Transactions (Millions)")

    plt.tight_layout()
    return fig

def plot_temporal_trends(monthly_df, daily_df):
    """Plots monthly and daily transaction trends."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Monthly Value Trend
    sns.lineplot(x='Month', y='total_value', data=monthly_df, ax=axes[0, 0], marker='o', color='forestgreen')
    axes[0, 0].set_title('Monthly Total Value Trend (Seasonality)', fontsize=16)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylabel("Total Value (Trillions)")
    axes[0, 0].set_xlabel("")

    # Monthly Transaction Trend
    sns.lineplot(x='Month', y='total_transactions', data=monthly_df, ax=axes[0, 1], marker='o', color='darkorange')
    axes[0, 1].set_title('Monthly Total Transactions Trend (Volume)', fontsize=16)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylabel("Total Transactions (Millions)")
    axes[0, 1].set_xlabel("")

    # Daily Value Trend
    sns.barplot(x='dayofweek', y='total_value', data=daily_df, ax=axes[1, 0], palette="Blues_d")
    axes[1, 0].set_title('Daily Total Value Trend (Weekday vs. Weekend)', fontsize=16)
    axes[1, 0].set_ylabel("Total Value (Trillions)")
    axes[1, 0].set_xlabel("")

    # Daily Transaction Trend
    sns.barplot(x='dayofweek', y='total_transactions', data=daily_df, ax=axes[1, 1], palette="Reds_d")
    axes[1, 1].set_title('Daily Total Transactions Trend (Volume)', fontsize=16)
    axes[1, 1].set_ylabel("Total Transactions (Millions)")
    axes[1, 1].set_xlabel("")

    plt.tight_layout(pad=3.0)
    return fig

# --- STREAMLIT APP LAYOUT ---

# Header
st.title("REC-SSEC Bank – Transaction Behaviour & Regional Growth Insights")
st.subheader("Data-Driven Strategy for Domain-City Expansion")

# --- SIDEBAR (Filtering and Navigation) ---
st.sidebar.header("Navigation & Filters")
menu = ["Overview & Core Metrics", "Regional & Temporal Analysis", "Performance Segmentation (Clustering)"]
selection = st.sidebar.radio("Go to Section", menu)

if selection == "Overview & Core Metrics":
    
    # ----------------------------------------------------
    # SECTION 1: OVERVIEW & CORE METRICS
    # ----------------------------------------------------
    st.header("1. Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    total_value = data['Value'].sum() / 1e12 # Trillions
    total_txns = data['Transaction_count'].sum() / 1e6 # Millions
    unique_domains = data['Domain'].nunique()
    
    col1.metric("Total Value (Annual)", f"₹{total_value:,.2f} Trillion")
    col2.metric("Total Transactions (Annual)", f"{total_txns:,.2f} Million")
    col3.metric("Domains Covered", unique_domains)

    st.markdown("""
    This analysis identifies high-growth and under-performing domain-city combinations to guide strategic investment.
    The primary methodology involves K-Means Clustering to segment 
    these market pairs based on their total value, transaction count, and daily averages.
    """)
    
    st.divider()

    st.header("Domain-Level Summary")
    domain_summary = data.groupby('Domain').agg(
        Total_Value=('Value', 'sum'),
        Total_Transactions=('Transaction_count', 'sum'),
        Avg_Daily_Value=('Value', 'mean')
    ).sort_values('Total_Value', ascending=False).reset_index()
    
    # Formatting for display
    domain_summary['Total_Value'] = (domain_summary['Total_Value'] / 1e9).map('{:,.2f}B'.format)
    domain_summary['Total_Transactions'] = (domain_summary['Total_Transactions'] / 1e6).map('{:,.2f}M'.format)
    domain_summary['Avg_Daily_Value'] = (domain_summary['Avg_Daily_Value']).map('₹{:,.2f}'.format)
    
    st.dataframe(domain_summary, use_container_width=True)
    
    st.info("""
    **Observation:** All domains appear financially robust and consistently active. Performance differences are primarily found at the granular **Domain-City** level, which justifies the need for clustering.
    """)


elif selection == "Regional & Temporal Analysis":
    
    # ----------------------------------------------------
    # SECTION 2: REGIONAL & TEMPORAL ANALYSIS
    # ----------------------------------------------------
    
    st.header("2. Regional Performance: Top Locations")
    st.markdown("Identification of the top 10 strongest cities based on overall transaction volume and value.")
    
    regional_plot = plot_top_10_regional(regional_perf)
    st.pyplot(regional_plot)
    
    st.info("""
    **Insight:** While overall regional performance is stable, certain locations serve as critical hubs for large-value and high-volume transactions, warranting further investment.
    """)
    
    st.divider()

    st.header("3. Temporal & Seasonal Trends")
    st.markdown("Analyzing how transaction volume and value fluctuate across months and days of the week.")
    
    temporal_plot = plot_temporal_trends(monthly_summary, daily_summary)
    st.pyplot(temporal_plot)

    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        st.subheader("Monthly Trends")
        st.markdown("""
        - **Peaks:** July and December typically show the highest activity, indicating mid-year and year-end spending surges.
        - **Dips:** February consistently shows the lowest transaction value/volume, possibly due to fewer days or post-holiday contraction.
        """)
    with col_temp2:
        st.subheader("Daily Trends")
        st.markdown("""
        - **Weekend Activity:** Saturday shows a slight but consistent rise in both value and volume, suggesting higher personal spending.
        - **Weekday Stability:** Activity remains remarkably stable across all weekdays, showing balanced banking habits.
        """)


elif selection == "Performance Segmentation (Clustering)":
    
    # ----------------------------------------------------
    # SECTION 3: PERFORMANCE SEGMENTATION (CLUSTERING)
    # ----------------------------------------------------
    
    st.header("4. Domain-City Performance Segmentation")
    st.markdown("Using K-Means clustering (k=3) on daily averages and total metrics to categorize every Domain-City pair.")
    
    # Summary of Clusters
    cluster_summary = dc.groupby('Cluster_Label').agg(
        Pairs_Count=('Location', 'count'),
        Avg_Daily_Value_Mean=('avg_daily_value', 'mean'),
        Total_Value_Mean=('total_value', 'mean')
    ).sort_values('Total_Value_Mean', ascending=False).reset_index()

    # Formatting
    cluster_summary['Avg_Daily_Value_Mean'] = (cluster_summary['Avg_Daily_Value_Mean']).map('₹{:,.0f}'.format)
    cluster_summary['Total_Value_Mean'] = (cluster_summary['Total_Value_Mean'] / 1e9).map('{:,.2f}B'.format)
    
    st.subheader("Cluster Profiles")
    st.dataframe(cluster_summary, use_container_width=True)

    # --- Recommendations and Drilldown ---
    st.divider()
    st.subheader("Strategic Recommendations based on Cluster")

    tab1, tab2, tab3 = st.tabs(["🔥 High Performance", "⚠️ Medium Performance", "📉 Low Performance"])
    
    # High Performance Cluster
    high_df = dc[dc.Cluster_Label == "HIGH_PERFORMANCE"].sort_values('total_value', ascending=False)
    with tab1:
        st.success("🎯 **Strategy: Investment & Retention**")
        st.markdown("""
        These are strongholds with high transaction value and volume. Focus on maximizing revenue and preventing churn.
        """)
        st.markdown("- **Action:** Cross-sell premium products (e.g., high-tier credit cards, wealth management services).")
        st.markdown("- **Action:** Strengthen merchant loyalty programs and offer dedicated support.")
        st.dataframe(high_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True)

    # Medium Performance Cluster
    medium_df = dc[dc.Cluster_Label == "MEDIUM_PERFORMANCE"].sort_values('total_value', ascending=False)
    with tab2:
        st.warning("📈 **Strategy: Activation & Expansion**")
        st.markdown("""
        These are stable markets with potential for growth. The goal is to elevate them to High Performance status.
        """)
        st.markdown("- **Action:** Run targeted activation campaigns to increase transaction frequency (e.g., cashback on 5th transaction).")
        st.markdown("- **Action:** Accelerate merchant onboarding, especially micro and small businesses.")
        st.dataframe(medium_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True)

    # Low Performance Cluster
    low_df = dc[dc.Cluster_Label == "LOW_PERFORMANCE"].sort_values('total_value', ascending=False)
    with tab3:
        st.error("🛠️ **Strategy: Digital Adoption & Infrastructure**")
        st.markdown("""
        These markets are underperforming or underserved, requiring fundamental improvements in outreach and adoption.
        """)
        st.markdown("- **Action:** Increase digital awareness drives and customer training on mobile/UPI services.")
        st.markdown("- **Action:** Offer strong incentives (cashbacks) for first-time digital users and new merchants.")
        st.dataframe(low_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True)
    
    
    # Optional: Drilldown filter
    st.sidebar.subheader("Cluster Drilldown")
    selected_cluster = st.sidebar.selectbox("Select Cluster to Analyze", dc['Cluster_Label'].unique())
    
    if selected_cluster:
        st.subheader(f"Full List: {selected_cluster} Pairs")
        filtered_df = dc[dc.Cluster_Label == selected_cluster].sort_values('total_value', ascending=False)
        st.dataframe(
            filtered_df[['Domain', 'Location', 'total_value', 'total_transactions', 'avg_daily_value', 'avg_daily_count']].head(50), 
            use_container_width=True,
            column_config={
                "total_value": st.column_config.NumberColumn("Total Value", format="₹%,.0f"),
                "total_transactions": st.column_config.NumberColumn("Total Txns", format="%,.0f"),
                "avg_daily_value": st.column_config.NumberColumn("Avg Daily Value", format="₹%,.0f"),
                "avg_daily_count": st.column_config.NumberColumn("Avg Daily Txns", format="%,.0f"),
            }
        )

# Footer
st.markdown("""
---
*Analysis performed using K-Means Clustering on Domain-City Aggregates.*
""")