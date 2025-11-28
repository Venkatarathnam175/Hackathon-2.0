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
    Loads the actual bank dataset from the uploaded files (handling split files),
    performs necessary data cleansing, aggregation, and clustering steps.
    """
    
    # ----------------------------------------------------
    # DATA LOADING (Using split CSV files to bypass GitHub size limit)
    # ----------------------------------------------------
    # Define the sequence of files to load
    file_names = [
        "bankdataset_part1.csv",
        "bankdataset_part2.csv",
        # IMPORTANT: Add more part file names here if you split it into more pieces (e.g., "bankdataset_part3.csv")
    ]
    
    data_frames = []
    
    for i, file_name in enumerate(file_names):
        # The first file should have the header, subsequent files should skip it
        header_row = 'infer' if i == 0 else None 
        
        try:
            # Attempt 1: UTF-8 encoding
            df_part = pd.read_csv(file_name, header=header_row, encoding='utf-8')
            
        except FileNotFoundError:
            st.warning(f"Data file not found: {file_name}. Please check file name and location in repository.")
            continue # Skip to the next file
            
        except UnicodeDecodeError:
            try:
                # Attempt 2: Latin-1 encoding (for common non-standard exports)
                df_part = pd.read_csv(file_name, header=header_row, encoding='latin-1')
            except Exception as e:
                st.error(f"Error loading {file_name}: Failed to load with UTF-8 and Latin-1 encoding.")
                continue

        except Exception as e:
            st.error(f"Critical error loading {file_name}. Details: {e}")
            continue

        # If it's not the first file, we need to assign the columns from the first successful load
        if i > 0 and data_frames:
            # Note: This ASSUMES bankdataset_part1.csv was the first one loaded.
            df_part.columns = data_frames[0].columns 
        
        # Append the successfully loaded part
        data_frames.append(df_part)
            
            
    if not data_frames:
        st.error("No data files were loaded successfully. Please ensure the files exist and have the correct format.")
        return None, None, None, None, None, None, None

    # Concatenate all parts into a single DataFrame
    data = pd.concat(data_frames, ignore_index=True)


    # Data Type Conversion and Cleaning
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Convert value and count columns to numeric, dropping any rows where this fails
    for col in ['Value', 'Transaction_count']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data.dropna(subset=[col], inplace=True) 

    # --- PREPROCESSING & AGGREGATION (from original notebook) ---
    data['Month'] = data['Date'].dt.to_period('M')
    data['dayofweek'] = data['Date'].dt.day_name()
    
    # 1. Domain-Level Summary
    domain_daily=data.groupby(['Domain','Date']).agg(daily_value=('Value','sum'),daily_count=('Transaction_count','sum')).reset_index()
    domain_summary= domain_daily.groupby(['Domain']).agg(
        avg_daily_value=('daily_value','mean'),
        avg_daily_count=('daily_count','mean'),
        total_value=('daily_value','sum'),
        total_transactions=('daily_count','sum'), 
        days_recorded=('Date','nunique')
    ).reset_index().sort_values('total_value', ascending=False)
    
    # 2. Regional Performance
    regional_perf = data.groupby(['Location']).agg(
        avg_txn_value=('Value','mean'),
        avg_txn_count=('Transaction_count','mean'),
        total_transactions=('Transaction_count','sum'),
        total_value=('Value','sum'),
        days_recorded=('Date','nunique')
    ).reset_index()
    
    # 3. Monthly & Daily Summary for Temporal Analysis
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
    
    # 4. Domain-Location Daily Aggregation for Clustering
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
        
    # 5. Domain and Location wise performance (for section 4)
    domain_loca_perf = data.groupby(['Domain','Location']).agg(
        avg_txn_value=('Value','mean'),
        avg_txn_count=('Transaction_count','mean'),
        total_value=('Value','sum'),
        total_transactions=('Transaction_count','sum'),
        days_recorded=('Date','nunique')
    ).reset_index()


    # --- K-MEANS CLUSTERING (k=3 based on notebook findings) ---
    X = dc[['avg_daily_value','avg_daily_count','total_value','total_transactions']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use k=3 as determined in the original analysis
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    dc['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map cluster labels (identifying clusters by their mean total performance)
    cluster_means = dc.groupby('Cluster')[['total_value', 'total_transactions']].mean().sum(axis=1).sort_values(ascending=False)
    
    cluster_names = {
        cluster_means.index[0]: "HIGH_PERFORMANCE",
        cluster_means.index[1]: "MEDIUM_PERFORMANCE",
        cluster_means.index[2]: "LOW_PERFORMANCE"
    }

    dc['Cluster_Label'] = dc['Cluster'].map(cluster_names)
    
    return data, domain_summary, regional_perf, monthly_summary, daily_summary, dc, domain_loca_perf

# Load all pre-processed dataframes
data, domain_summary, regional_perf, monthly_summary, daily_summary, dc, domain_loca_perf = load_and_process_data()

# Check if data loading was successful
if data is None:
    st.stop()


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
    axes[0].set_ylabel("Total Value") 
    axes[0].ticklabel_format(style='plain', axis='y') 

    sns.barplot(x='Location', y='total_transactions', data=top10_count, ax=axes[1], palette="magma")
    axes[1].set_title('Top 10 Locations by Total Transactions (Volume)', fontsize=16)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total Transactions")
    axes[1].ticklabel_format(style='plain', axis='y')

    plt.tight_layout()
    return fig

def plot_temporal_trends(monthly_df, daily_df):
    """Plots monthly and daily transaction trends."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Monthly Value Trend
    sns.lineplot(x='Month', y='total_value', data=monthly_df, ax=axes[0, 0], marker='o', color='forestgreen')
    axes[0, 0].set_title('Monthly Total Value Trend (Seasonality)', fontsize=16)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylabel("Total Value")
    axes[0, 0].set_xlabel("")

    # Monthly Transaction Trend
    sns.lineplot(x='Month', y='total_transactions', data=monthly_df, ax=axes[0, 1], marker='o', color='darkorange')
    axes[0, 1].set_title('Monthly Total Transactions Trend (Volume)', fontsize=16)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylabel("Total Transactions")
    axes[0, 1].set_xlabel("")

    # Daily Value Trend
    sns.barplot(x='dayofweek', y='total_value', data=daily_df, ax=axes[1, 0], palette="Blues_d")
    axes[1, 0].set_title('Daily Total Value Trend (Weekday vs. Weekend)', fontsize=16)
    axes[1, 0].set_ylabel("Total Value")
    axes[1, 0].set_xlabel("")

    # Daily Transaction Trend
    sns.barplot(x='dayofweek', y='total_transactions', data=daily_df, ax=axes[1, 1], palette="Reds_d")
    axes[1, 1].set_title('Daily Total Transactions Trend (Volume)', fontsize=16)
    axes[1, 1].set_ylabel("Total Transactions")
    axes[1, 1].set_xlabel("")

    plt.tight_layout(pad=3.0)
    return fig
    
def plot_domain_location_matrix(df):
    """Plots a heatmap of Total Value by Domain and Location."""
    # Pivot for Heatmap visualization
    pivot_table = df.pivot_table(
        index='Location', 
        columns='Domain', 
        values='total_value', 
        aggfunc='sum'
    )
    
    plt.figure(figsize=(20, 15))
    sns.heatmap(
        pivot_table, 
        cmap="YlGnBu", 
        annot=False, 
        fmt=".1f",
        linewidths=.5, 
        cbar_kws={'label': 'Total Value'}
    )
    plt.title('Domain-Location Performance Matrix (Total Value)', fontsize=18)
    plt.xlabel('Domain')
    plt.ylabel('Location')
    plt.tight_layout()
    return plt.gcf()


# --- STREAMLIT APP LAYOUT ---

# Header
st.title("REC-SSEC Bank – Transaction Behaviour & Regional Growth Insights")
st.subheader("Data-Driven Strategy for Domain-City Expansion")

# --- SIDEBAR (Filtering and Navigation) ---
st.sidebar.header("Navigation & Filters")
menu = [
    "1. Overview",
    "2. Domain-Level Performance",
    "3. Regional-Wise Performance",
    "4. Domain and Location Wise Performance",
    "5. Temporal and Seasonal Analysis",
    "6. Clustering and Its Results"
]
selection = st.sidebar.radio("Go to Section", menu)

# --- NAVIGATION IMPLEMENTATION ---

if selection == "1. Overview":
    
    # ----------------------------------------------------
    # SECTION 1: OVERVIEW & CORE METRICS
    # ----------------------------------------------------
    st.header("1. Executive Summary")
    
    col1, col2, col3 = st.columns(3)
    
    # Using actual data for metrics, displayed in Billions and Millions
    total_value = data['Value'].sum() / 1e9 # Billions
    total_txns = data['Transaction_count'].sum() / 1e6 # Millions
    unique_domains = data['Domain'].nunique()
    
    col1.metric("Total Value (Annual)", f"₹{total_value:,.2f} Billion")
    col2.metric("Total Transactions (Annual)", f"{total_txns:,.2f} Million")
    col3.metric("Domains Covered", unique_domains)

    st.markdown("""
    This analysis identifies high-growth and under-performing domain-city combinations to guide strategic investment.
    The primary methodology involves K-Means Clustering to segment 
    these market pairs based on their total value, transaction count, and daily averages.
    """)
    
    st.divider()
    st.markdown("Use the navigation panel on the left to explore the detailed analysis.")


elif selection == "2. Domain-Level Performance":
    
    # ----------------------------------------------------
    # SECTION 2: DOMAIN-LEVEL PERFORMANCE
    # ----------------------------------------------------
    st.header("2. Domain-Level Performance")
    st.markdown("Summary of aggregated transaction value and volume across all bank domains.")

    domain_disp = domain_summary.copy()
    # Formatting for display
    domain_disp['total_value'] = (domain_disp['total_value'] / 1e9).map('₹{:,.2f}B'.format)
    domain_disp['total_transactions'] = (domain_disp['total_transactions'] / 1e6).map('{:,.2f}M'.format)
    domain_disp['avg_daily_value'] = (domain_disp['avg_daily_value']).map('₹{:,.2f}'.format)
    domain_disp['avg_daily_count'] = (domain_disp['avg_daily_count']).map('{:,.0f}'.format)
    
    st.dataframe(
        domain_disp, 
        use_container_width=True,
        column_order=['Domain', 'total_value', 'total_transactions', 'avg_daily_value', 'avg_daily_count', 'days_recorded']
    )
    
    st.subheader("Observations")
    # Using actual data for observations
    min_daily_value = domain_summary['avg_daily_value'].min()
    max_daily_value = domain_summary['avg_daily_value'].max()
    min_daily_count = domain_summary['avg_daily_count'].min()
    max_daily_count = domain_summary['avg_daily_count'].max()

    st.markdown(f"""
    - **Daily Revenue:** All Domains show nearly identical average daily revenue, ranging from **₹{min_daily_value:,.2f} to ₹{max_daily_value:,.2f}**.
    - **Daily Transactions:** All Domains show highly similar average daily transaction volumes, ranging from **{min_daily_count:,.0f} to {max_daily_count:,.0f}** transactions per day.
    - All Domains are making consistent transactions.
    - This homogeneity suggests that the bank has a **well-diversified and stable transaction portfolio**, and is **not overly dependent on any single domain** for revenue or transaction volume.
    """)


elif selection == "3. Regional-Wise Performance":
    
    # ----------------------------------------------------
    # SECTION 3: REGIONAL-WISE PERFORMANCE
    # ----------------------------------------------------
    st.header("3. Regional-Wise Performance")
    st.markdown("Identification of the top 10 strongest cities based on overall transaction volume and value.")
    
    regional_plot = plot_top_10_regional(regional_perf)
    st.pyplot(regional_plot)
    
    st.info("""
    **Insight:** While overall regional performance is highly consistent, certain locations serve as critical hubs for transactions. The high degree of uniformity suggests strong operational consistency and balanced merchant penetration across the bank's network.
    """)

elif selection == "4. Domain and Location Wise Performance":
    
    # ----------------------------------------------------
    # SECTION 4: DOMAIN AND LOCATION WISE PERFORMANCE
    # ----------------------------------------------------
    st.header("4. Domain and Location Wise Performance")
    st.markdown("A deep dive into the performance of every combination of Domain and City, highlighting where specific domains thrive.")

    st.subheader("Top 10 Domain-City Pairs by Total Value")
    top_pairs = domain_loca_perf.nlargest(10, 'total_value').copy()
    
    # Formatting for display
    top_pairs['total_value'] = (top_pairs['total_value'] / 1e6).map('₹{:,.2f}M'.format)
    top_pairs['total_transactions'] = (top_pairs['total_transactions'] / 1e3).map('{:,.2f}K'.format)
    
    st.dataframe(
        top_pairs[['Domain', 'Location', 'total_value', 'total_transactions']], 
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Performance Matrix (Heatmap)")
    st.markdown("This heatmap visually identifies the strongest Domain-Location pairs based on total transaction value.")
    
    heatmap_fig = plot_domain_location_matrix(domain_loca_perf)
    st.pyplot(heatmap_fig)

    st.info("The strongest individual Domain-City pairs are the key targets for strategic marketing and partnership deepening.")


elif selection == "5. Temporal and Seasonal Analysis":
    
    # ----------------------------------------------------
    # SECTION 5: TEMPORAL AND SEASONAL ANALYSIS
    # ----------------------------------------------------
    st.header("5. Temporal and Seasonal Analysis")
    st.markdown("Analyzing how transaction volume and value fluctuate across months and days of the week.")
    
    temporal_plot = plot_temporal_trends(monthly_summary, daily_summary)
    st.pyplot(temporal_plot)

    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        st.subheader("Monthly Trends")
        st.markdown("""
        - **Peaks:** The data clearly shows peak activity months, indicating periods of high consumer spending (e.g., mid-year and year-end surges).
        - **Dips:** Consistent low transaction value/volume months suggest periods of post-holiday contraction or low commercial activity.
        """)
    with col_temp2:
        st.subheader("Daily Trends")
        st.markdown("""
        - **Weekend Activity:** There is a distinct trend showing a rise in activity on Saturdays, suggesting higher personal spending.
        - **Weekday Stability:** Activity remains largely stable across all weekdays, showing balanced banking habits.
        """)


elif selection == "6. Clustering and Its Results":
    
    # ----------------------------------------------------
    # SECTION 6: CLUSTERING AND ITS RESULTS
    # ----------------------------------------------------
    
    st.header("6. Clustering and Its Results")
    st.markdown("Using K-Means clustering (k=3) on daily averages and total metrics to categorize every Domain-City pair into performance segments.")
    
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
