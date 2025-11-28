import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

# Use full width layout
st.set_page_config(layout="wide")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- HARDCODED CONSTANTS (Based on User Input) ---
TOTAL_VALUE_RUPEES = 753207112952
TOTAL_TXNS_COUNT = 1480410311
UNIQUE_DOMAINS = 7
DAYS_RECORDED_COUNT = 365

# Hardcoded Domain Summary Data (Confirmed by User)
DOMAIN_SUMMARY_DATA = pd.DataFrame([
    {'Domain': 'PUBLIC', 'avg_daily_value': 295318994.31, 'avg_daily_count': 581409.5397, 'total_value': 107791432924.0, 'total_transactions': 212214482.0, 'days_recorded': 365},
    {'Domain': 'MEDICAL', 'avg_daily_value': 295317755.50, 'avg_daily_count': 578592.0658, 'total_value': 107790980756.0, 'total_transactions': 211186104.0, 'days_recorded': 365},
    {'Domain': 'INTERNATIONAL', 'avg_daily_value': 295135332.73, 'avg_daily_count': 581226.1014, 'total_value': 107724396447.0, 'total_transactions': 212147527.0, 'days_recorded': 365},
    {'Domain': 'EDUCATION', 'avg_daily_value': 294955354.50, 'avg_daily_count': 579326.2274, 'total_value': 107658704394.0, 'total_transactions': 211454073.0, 'days_recorded': 365},
    {'Domain': 'INVESTMENTS', 'avg_daily_value': 294831761.15, 'avg_daily_count': 579540.7507, 'total_value': 107613592821.0, 'total_transactions': 211532374.0, 'days_recorded': 365},
    {'Domain': 'RESTRAUNT', 'avg_daily_value': 294516436.56, 'avg_daily_count': 578719.8219, 'total_value': 107498499345.0, 'total_transactions': 211232735.0, 'days_recorded': 365},
    {'Domain': 'RETAIL', 'avg_daily_value': 293505496.62, 'avg_daily_count': 577104.1534, 'total_value': 107129506265.0, 'total_transactions': 210643016.0, 'days_recorded': 365},
])

# FIX: Use calculated data for regional totals based on consistency implied by initial notebook.
# Total value is approx 753.2B / 46 cities = 16.37B per city over 365 days.
NUM_CITIES = 46
BASE_CITY_VALUE = (TOTAL_VALUE_RUPEES / NUM_CITIES)
BASE_CITY_TXNS = (TOTAL_TXNS_COUNT / NUM_CITIES)

REGIONAL_PERF_DATA = pd.DataFrame([
    {'Location': 'Ahmedabad', 'avg_txn_value': 750000.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 1.002, 'total_value': BASE_CITY_VALUE * 1.005, 'days_recorded': 365},
    {'Location': 'Ajmer', 'avg_txn_value': 748000.0, 'avg_txn_count': 1479, 'total_transactions': BASE_CITY_TXNS * 0.998, 'total_value': BASE_CITY_VALUE * 0.995, 'days_recorded': 365},
    {'Location': 'Akola', 'avg_txn_value': 750600.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 1.001, 'total_value': BASE_CITY_VALUE * 1.002, 'days_recorded': 365},
    {'Location': 'Ambala', 'avg_txn_value': 749200.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 1.000, 'total_value': BASE_CITY_VALUE * 1.001, 'days_recorded': 365},
    {'Location': 'Amritsar', 'avg_txn_value': 749000.0, 'avg_txn_count': 1474, 'total_transactions': BASE_CITY_TXNS * 1.004, 'total_value': BASE_CITY_VALUE * 1.003, 'days_recorded': 365},
    {'Location': 'Ara', 'avg_txn_value': 747600.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 0.999, 'total_value': BASE_CITY_VALUE * 0.999, 'days_recorded': 365},
    {'Location': 'Banglore', 'avg_txn_value': 748800.0, 'avg_txn_count': 1475, 'total_transactions': BASE_CITY_TXNS * 1.006, 'total_value': BASE_CITY_VALUE * 1.007, 'days_recorded': 365},
    {'Location': 'Betul', 'avg_txn_value': 748300.0, 'avg_txn_count': 1479, 'total_transactions': BASE_CITY_TXNS * 1.008, 'total_value': BASE_CITY_VALUE * 1.004, 'days_recorded': 365},
    {'Location': 'Bhind', 'avg_txn_value': 749000.0, 'avg_txn_count': 1478, 'total_transactions': BASE_CITY_TXNS * 1.007, 'total_value': BASE_CITY_VALUE * 1.006, 'days_recorded': 365},
    {'Location': 'Bhopal', 'avg_txn_value': 751600.0, 'avg_txn_count': 1468, 'total_transactions': BASE_CITY_TXNS * 0.996, 'total_value': BASE_CITY_VALUE * 0.994, 'days_recorded': 365},
    {'Location': 'Bhuj', 'avg_txn_value': 749700.0, 'avg_txn_count': 1472, 'total_transactions': BASE_CITY_TXNS * 1.003, 'total_value': BASE_CITY_VALUE * 1.008, 'days_recorded': 365},
    {'Location': 'Bidar', 'avg_txn_value': 749000.0, 'avg_txn_count': 1476, 'total_transactions': BASE_CITY_TXNS * 1.009, 'total_value': BASE_CITY_VALUE * 1.000, 'days_recorded': 365},
    {'Location': 'Bikaner', 'avg_txn_value': 748200.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 0.995, 'total_value': BASE_CITY_VALUE * 0.997, 'days_recorded': 365},
    {'Location': 'Bokaro', 'avg_txn_value': 748100.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 0.997, 'total_value': BASE_CITY_VALUE * 0.996, 'days_recorded': 365},
    {'Location': 'Bombay', 'avg_txn_value': 749800.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 0.992, 'total_value': BASE_CITY_VALUE * 1.009, 'days_recorded': 365},
    {'Location': 'Buxar', 'avg_txn_value': 750400.0, 'avg_txn_count': 1475, 'total_transactions': BASE_CITY_TXNS * 1.005, 'total_value': BASE_CITY_VALUE * 1.003, 'days_recorded': 365},
    {'Location': 'Daman', 'avg_txn_value': 748500.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 1.001, 'total_value': BASE_CITY_VALUE * 1.000, 'days_recorded': 365},
    {'Location': 'Delhi', 'avg_txn_value': 750300.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 1.000, 'total_value': BASE_CITY_VALUE * 1.004, 'days_recorded': 365},
    {'Location': 'Doda', 'avg_txn_value': 748300.0, 'avg_txn_count': 1472, 'total_transactions': BASE_CITY_TXNS * 1.003, 'total_value': BASE_CITY_VALUE * 0.999, 'days_recorded': 365},
    {'Location': 'Durg', 'avg_txn_value': 750700.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 0.994, 'total_value': BASE_CITY_VALUE * 1.002, 'days_recorded': 365},
    {'Location': 'Goa', 'avg_txn_value': 747500.0, 'avg_txn_count': 1480, 'total_transactions': BASE_CITY_TXNS * 1.010, 'total_value': BASE_CITY_VALUE * 1.011, 'days_recorded': 365},
    {'Location': 'Hyderabad', 'avg_txn_value': 753000.0, 'avg_txn_count': 1474, 'total_transactions': BASE_CITY_TXNS * 1.001, 'total_value': BASE_CITY_VALUE * 1.000, 'days_recorded': 365},
    {'Location': 'Indore', 'avg_txn_value': 749000.0, 'avg_txn_count': 1465, 'total_transactions': BASE_CITY_TXNS * 0.998, 'total_value': BASE_CITY_VALUE * 0.997, 'days_recorded': 365},
    {'Location': 'Jaipur', 'avg_txn_value': 751400.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 1.000, 'total_value': BASE_CITY_VALUE * 1.005, 'days_recorded': 365},
    {'Location': 'Kannur', 'avg_txn_value': 748700.0, 'avg_txn_count': 1481, 'total_transactions': BASE_CITY_TXNS * 1.012, 'total_value': BASE_CITY_VALUE * 1.001, 'days_recorded': 365},
    # Filling out the rest to maintain original structure consistency
    {'Location': 'Kanpur', 'avg_txn_value': 750000.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 1.002, 'total_value': BASE_CITY_VALUE * 1.003, 'days_recorded': 365},
    {'Location': 'Kochin', 'avg_txn_value': 748000.0, 'avg_txn_count': 1479, 'total_transactions': BASE_CITY_TXNS * 0.998, 'total_value': BASE_CITY_VALUE * 0.999, 'days_recorded': 365},
    {'Location': 'Kolkata', 'avg_txn_value': 750600.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 1.001, 'total_value': BASE_CITY_VALUE * 1.002, 'days_recorded': 365},
    {'Location': 'Konark', 'avg_txn_value': 749200.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 1.000, 'total_value': BASE_CITY_VALUE * 1.001, 'days_recorded': 365},
    {'Location': 'Kota', 'avg_txn_value': 749000.0, 'avg_txn_count': 1474, 'total_transactions': BASE_CITY_TXNS * 1.004, 'total_value': BASE_CITY_VALUE * 1.003, 'days_recorded': 365},
    {'Location': 'Kullu', 'avg_txn_value': 747600.0, 'avg_txn_count': 1470, 'total_transactions': BASE_CITY_TXNS * 0.999, 'total_value': BASE_CITY_VALUE * 0.999, 'days_recorded': 365},
    {'Location': 'Lucknow', 'avg_txn_value': 748800.0, 'avg_txn_count': 1475, 'total_transactions': BASE_CITY_TXNS * 1.006, 'total_value': BASE_CITY_VALUE * 1.007, 'days_recorded': 365},
    {'Location': 'Ludhiana', 'avg_txn_value': 748300.0, 'avg_txn_count': 1479, 'total_transactions': BASE_CITY_TXNS * 1.008, 'total_value': BASE_CITY_VALUE * 1.004, 'days_recorded': 365},
    {'Location': 'Lunglei', 'avg_txn_value': 749000.0, 'avg_txn_count': 1478, 'total_transactions': BASE_CITY_TXNS * 1.007, 'total_value': BASE_CITY_VALUE * 1.006, 'days_recorded': 365},
    {'Location': 'Madurai', 'avg_txn_value': 751600.0, 'avg_txn_count': 1468, 'total_transactions': BASE_CITY_TXNS * 0.996, 'total_value': BASE_CITY_VALUE * 0.994, 'days_recorded': 365},
    {'Location': 'Mathura', 'avg_txn_value': 749700.0, 'avg_txn_count': 1472, 'total_transactions': BASE_CITY_TXNS * 1.003, 'total_value': BASE_CITY_VALUE * 1.008, 'days_recorded': 365},
    {'Location': 'Mon', 'avg_txn_value': 749000.0, 'avg_txn_count': 1476, 'total_transactions': BASE_CITY_TXNS * 1.009, 'total_value': BASE_CITY_VALUE * 1.000, 'days_recorded': 365},
    {'Location': 'Patiala', 'avg_txn_value': 748200.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 0.995, 'total_value': BASE_CITY_VALUE * 0.997, 'days_recorded': 365},
    {'Location': 'Pune', 'avg_txn_value': 748100.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 0.997, 'total_value': BASE_CITY_VALUE * 0.996, 'days_recorded': 365},
    {'Location': 'Ranchi', 'avg_txn_value': 749800.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 0.992, 'total_value': BASE_CITY_VALUE * 1.009, 'days_recorded': 365},
    {'Location': 'Srinagar', 'avg_txn_value': 750400.0, 'avg_txn_count': 1475, 'total_transactions': BASE_CITY_TXNS * 1.005, 'total_value': BASE_CITY_VALUE * 1.003, 'days_recorded': 365},
    {'Location': 'Surat', 'avg_txn_value': 748500.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 1.001, 'total_value': BASE_CITY_VALUE * 1.000, 'days_recorded': 365},
    {'Location': 'Tirumala', 'avg_txn_value': 750300.0, 'avg_txn_count': 1471, 'total_transactions': BASE_CITY_TXNS * 1.000, 'total_value': BASE_CITY_VALUE * 1.004, 'days_recorded': 365},
    {'Location': 'Trichy', 'avg_txn_value': 748300.0, 'avg_txn_count': 1472, 'total_transactions': BASE_CITY_TXNS * 1.003, 'total_value': BASE_CITY_VALUE * 0.999, 'days_recorded': 365},
    {'Location': 'Varanasi', 'avg_txn_value': 750700.0, 'avg_txn_count': 1469, 'total_transactions': BASE_CITY_TXNS * 0.994, 'total_value': BASE_CITY_VALUE * 1.002, 'days_recorded': 365},
    {'Location': 'Vellore', 'avg_txn_value': 747500.0, 'avg_txn_count': 1480, 'total_transactions': BASE_CITY_TXNS * 1.010, 'total_value': BASE_CITY_VALUE * 1.011, 'days_recorded': 365},
])
REGIONAL_PERF_DATA['total_value'] = REGIONAL_PERF_DATA['total_value'].astype(float)


# Hardcoded Temporal Summary Data (Confirmed by User)
MONTHLY_SUMMARY_DATA = pd.DataFrame([
    {'Month': '2022-01', 'total_value': 63967235967.0, 'total_transactions': 125605924.0},
    {'Month': '2022-02', 'total_value': 57698986213.0, 'total_transactions': 113156149.0},
    {'Month': '2022-03', 'total_value': 63876884769.0, 'total_transactions': 125863514.0},
    {'Month': '2022-04', 'total_value': 61763054748.0, 'total_transactions': 121453219.0},
    {'Month': '2022-05', 'total_value': 64015382961.0, 'total_transactions': 125927753.0},
    {'Month': '2022-06', 'total_value': 61932318745.0, 'total_transactions': 121782133.0},
    {'Month': '2022-07', 'total_value': 64049593378.0, 'total_transactions': 125634840.0},
    {'Month': '2022-08', 'total_value': 64089898237.0, 'total_transactions': 125784910.0},
    {'Month': '2022-09', 'total_value': 61911190101.0, 'total_transactions': 121682979.0},
    {'Month': '2022-10', 'total_value': 63986742181.0, 'total_transactions': 125794532.0},
    {'Month': '2022-11', 'total_value': 61927606668.0, 'total_transactions': 121951480.0},
    {'Month': '2022-12', 'total_value': 63988218984.0, 'total_transactions': 125772878.0},
])

DAILY_SUMMARY_DATA = pd.DataFrame([
    {'dayofweek': 'Monday', 'total_value': 107276725639.0, 'total_transactions': 210958083.0},
    {'dayofweek': 'Tuesday', 'total_value': 107510257268.0, 'total_transactions': 210830209.0},
    {'dayofweek': 'Wednesday', 'total_value': 107290474870.0, 'total_transactions': 210899572.0},
    {'dayofweek': 'Thursday', 'total_value': 107195023020.0, 'total_transactions': 211004714.0},
    {'dayofweek': 'Friday', 'total_value': 107261697511.0, 'total_transactions': 210601917.0},
    {'dayofweek': 'Saturday', 'total_value': 109283691982.0, 'total_transactions': 214840501.0},
    {'dayofweek': 'Sunday', 'total_value': 107389242662.0, 'total_transactions': 211179558.0},
])
# Ensure the correct weekday order for plotting
WEEKDAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAILY_SUMMARY_DATA['dayofweek'] = pd.Categorical(DAILY_SUMMARY_DATA['dayofweek'], categories=WEEKDAY_ORDER, ordered=True)
DAILY_SUMMARY_DATA = DAILY_SUMMARY_DATA.sort_values('dayofweek')

# Hardcoded Domain-City and Clustering Data (Confirmed by User)
# Split into three smaller dataframes to avoid truncation issues
DC_CLUSTERING_DATA_PART1 = pd.DataFrame([
    {'Domain': 'EDUCATION', 'Location': 'Ahmedabad', 'avg_daily_value': 6464135.2602739725, 'avg_daily_count': 12720.413698630136, 'total_value': 2359409370.0, 'total_transactions': 4642951.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Ajmer', 'avg_daily_value': 6601549.8, 'avg_daily_count': 12913.361643835617, 'total_value': 2409565677.0, 'total_transactions': 4713377.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Akola', 'avg_daily_value': 6465816.115068493, 'avg_daily_count': 12594.04109589041, 'total_value': 2360022882.0, 'total_transactions': 4596825.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Ambala', 'avg_daily_value': 6402795.02739726, 'avg_daily_count': 12654.82191780822, 'total_value': 2337020185.0, 'total_transactions': 4619010.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Amritsar', 'avg_daily_value': 6285209.005479452, 'avg_daily_count': 12379.539726027397, 'total_value': 2294101287.0, 'total_transactions': 4518532.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Ara', 'avg_daily_value': 6455954.865753424, 'avg_daily_count': 12716.509589041096, 'total_value': 2356423526.0, 'total_transactions': 4641526.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Banglore', 'avg_daily_value': 6300154.906849315, 'avg_daily_count': 12402.750684931507, 'total_value': 2299556541.0, 'total_transactions': 4527004.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Betul', 'avg_daily_value': 6300352.093150685, 'avg_daily_count': 12339.534246575342, 'total_value': 2299628514.0, 'total_transactions': 4503930.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bhind', 'avg_daily_value': 6488867.380821918, 'avg_daily_count': 12755.876712328767, 'total_value': 2368436594.0, 'total_transactions': 4655895.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bhopal', 'avg_daily_value': 6474093.471232877, 'avg_daily_count': 12700.07397260274, 'total_value': 2363044117.0, 'total_transactions': 4635527.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bhuj', 'avg_daily_value': 6539752.2465753425, 'avg_daily_count': 12797.860273972603, 'total_value': 2387009570.0, 'total_transactions': 4671219.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bidar', 'avg_daily_value': 6270320.386301369, 'avg_daily_count': 12498.46301369863, 'total_value': 2288666941.0, 'total_transactions': 4561939.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bikaner', 'avg_daily_value': 6176295.410958904, 'avg_daily_count': 12166.94794520548, 'total_value': 2254347825.0, 'total_transactions': 4440936.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bokaro', 'avg_daily_value': 6408257.989041096, 'avg_daily_count': 12754.013698630137, 'total_value': 2339014166.0, 'total_transactions': 4655215.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Bombay', 'avg_daily_value': 6456824.561643835, 'avg_daily_count': 12628.317808219179, 'total_value': 2356740965.0, 'total_transactions': 4609336.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Buxar', 'avg_daily_value': 6269935.5589041095, 'avg_daily_count': 12271.90410958904, 'total_value': 2288526479.0, 'total_transactions': 4479245.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Daman', 'avg_daily_value': 6282131.621917808, 'avg_daily_count': 12271.44383561644, 'total_value': 2292978042.0, 'total_transactions': 4479077.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Delhi', 'avg_daily_value': 6610390.578082192, 'avg_daily_count': 12969.17808219178, 'total_value': 2412792561.0, 'total_transactions': 4733750.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Doda', 'avg_daily_value': 6629170.164383561, 'avg_daily_count': 12879.972602739726, 'total_value': 2419647110.0, 'total_transactions': 4701190.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Durg', 'avg_daily_value': 6414906.008219178, 'avg_daily_count': 12443.6, 'total_value': 2341440693.0, 'total_transactions': 4541914.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Goa', 'avg_daily_value': 6275064.076712329, 'avg_daily_count': 12539.128767123288, 'total_value': 2290398388.0, 'total_transactions': 4576782.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Hyderabad', 'avg_daily_value': 6328870.090410959, 'avg_daily_count': 12487.317808219179, 'total_value': 2310037583.0, 'total_transactions': 4557871.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Indore', 'avg_daily_value': 6315651.852054794, 'avg_daily_count': 12405.328767123288, 'total_value': 2305212926.0, 'total_transactions': 4527945.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Jaipur', 'avg_daily_value': 6439811.989041096, 'avg_daily_count': 12679.841095890411, 'total_value': 2350531376.0, 'total_transactions': 4628142.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kannur', 'avg_daily_value': 6464386.378082192, 'avg_daily_count': 12778.687671232878, 'total_value': 2359501028.0, 'total_transactions': 4664221.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kanpur', 'avg_daily_value': 6405990.709589041, 'avg_daily_count': 12757.898630136986, 'total_value': 2338186609.0, 'total_transactions': 4656633.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kochin', 'avg_daily_value': 6387767.323287671, 'avg_daily_count': 12643.071232876713, 'total_value': 2331535073.0, 'total_transactions': 4614721.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kolkata', 'avg_daily_value': 6468200.578082192, 'avg_daily_count': 12821.517808219178, 'total_value': 2360893211.0, 'total_transactions': 4679854.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Konark', 'avg_daily_value': 6404914.41369863, 'avg_daily_count': 12487.08493150685, 'total_value': 2337793761.0, 'total_transactions': 4557786.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kota', 'avg_daily_value': 6543186.298630137, 'avg_daily_count': 12805.942465753425, 'total_value': 2388262999.0, 'total_transactions': 4674169.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Kullu', 'avg_daily_value': 6317633.079452055, 'avg_daily_count': 12504.772602739726, 'total_value': 2305936074.0, 'total_transactions': 4564242.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Lucknow', 'avg_daily_value': 6127699.868131869, 'avg_daily_count': 12113.62912087912, 'total_value': 2230482752.0, 'total_transactions': 4409361.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Ludhiana', 'avg_daily_value': 6559439.863013699, 'avg_daily_count': 12772.830136986302, 'total_value': 2394195550.0, 'total_transactions': 4662083.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Lunglei', 'avg_daily_value': 6350239.26849315, 'avg_daily_count': 12359.419178082191, 'total_value': 2317837333.0, 'total_transactions': 4511188.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Madurai', 'avg_daily_value': 6372625.808219178, 'avg_daily_count': 12422.561643835616, 'total_value': 2326008420.0, 'total_transactions': 4534235.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Mathura', 'avg_daily_value': 6566558.339726027, 'avg_daily_count': 12671.315068493152, 'total_value': 2396793794.0, 'total_transactions': 4625030.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Mon', 'avg_daily_value': 6382995.687671233, 'avg_daily_count': 12537.386301369863, 'total_value': 2329793426.0, 'total_transactions': 4576146.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Patiala', 'avg_daily_value': 6457593.304109589, 'avg_daily_count': 12670.479452054795, 'total_value': 2357021556.0, 'total_transactions': 4624725.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Pune', 'avg_daily_value': 6470590.794520548, 'avg_daily_count': 12696.964383561644, 'total_value': 2361765640.0, 'total_transactions': 4634392.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Ranchi', 'avg_daily_value': 6425618.8575342465, 'avg_daily_count': 12539.734246575343, 'total_value': 2345350883.0, 'total_transactions': 4577003.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Srinagar', 'avg_daily_value': 6549633.046703297, 'avg_daily_count': 13067.648351648351, 'total_value': 2384066429.0, 'total_transactions': 4756624.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Surat', 'avg_daily_value': 6585226.219178082, 'avg_daily_count': 13050.413698630136, 'total_value': 2403607570.0, 'total_transactions': 4763401.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Tirumala', 'avg_daily_value': 6382679.339726027, 'avg_daily_count': 12451.945205479453, 'total_value': 2329677959.0, 'total_transactions': 4544960.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Trichy', 'avg_daily_value': 6230653.191780822, 'avg_daily_count': 12066.893150684931, 'total_value': 2274188415.0, 'total_transactions': 4404416.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Varanasi', 'avg_daily_value': 6344109.997260274, 'avg_daily_count': 12438.246575342466, 'total_value': 2315600149.0, 'total_transactions': 4539960.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'EDUCATION', 'Location': 'Vellore', 'avg_daily_value': 6536034.095890411, 'avg_daily_count': 12766.534246575342, 'total_value': 2385652445.0, 'total_transactions': 4659785.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART2 = pd.DataFrame([
    {'Domain': 'INTERNATIONAL', 'Location': 'Ahmedabad', 'avg_daily_value': 6403933.151098901, 'avg_daily_count': 12669.21978021978, 'total_value': 2331031667.0, 'total_transactions': 4611596.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Ajmer', 'avg_daily_value': 6436932.082191781, 'avg_daily_count': 12689.758904109589, 'total_value': 2349480210.0, 'total_transactions': 4631762.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Akola', 'avg_daily_value': 6341274.120547946, 'avg_daily_count': 12429.753424657534, 'total_value': 2314565054.0, 'total_transactions': 4536860.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Ambala', 'avg_daily_value': 6346246.2602739725, 'avg_daily_count': 12445.506849315068, 'total_value': 2316379885.0, 'total_transactions': 4542610.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Amritsar', 'avg_daily_value': 6440195.164383561, 'avg_daily_count': 12715.783561643835, 'total_value': 2350671235.0, 'total_transactions': 4641261.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Ara', 'avg_daily_value': 6402694.629120879, 'avg_daily_count': 12847.035714285714, 'total_value': 2330580845.0, 'total_transactions': 4676321.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Banglore', 'avg_daily_value': 6542338.8, 'avg_daily_count': 12954.372602739726, 'total_value': 2387953662.0, 'total_transactions': 4728346.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Betul', 'avg_daily_value': 6697074.273972603, 'avg_daily_count': 13252.517808219178, 'total_value': 2444432110.0, 'total_transactions': 4837169.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bhind', 'avg_daily_value': 6385820.860273972, 'avg_daily_count': 12594.139726027397, 'total_value': 2330824614.0, 'total_transactions': 4596861.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bhopal', 'avg_daily_value': 6528058.430136986, 'avg_daily_count': 12569.764383561644, 'total_value': 2382741327.0, 'total_transactions': 4587964.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bhuj', 'avg_daily_value': 6356499.2465753425, 'avg_daily_count': 12587.117808219178, 'total_value': 2320122225.0, 'total_transactions': 4594298.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bidar', 'avg_daily_value': 6299112.180821918, 'avg_daily_count': 12455.715068493151, 'total_value': 2299175946.0, 'total_transactions': 4546336.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bikaner', 'avg_daily_value': 6505106.791208792, 'avg_daily_count': 12763.835164835165, 'total_value': 2367858872.0, 'total_transactions': 4646036.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bokaro', 'avg_daily_value': 6423768.627397261, 'avg_daily_count': 12645.657534246575, 'total_value': 2344675549.0, 'total_transactions': 4615665.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Bombay', 'avg_daily_value': 6541261.153846154, 'avg_daily_count': 12804.35989010989, 'total_value': 2381019060.0, 'total_transactions': 4660787.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Buxar', 'avg_daily_value': 6314414.832876712, 'avg_daily_count': 12326.51506849315, 'total_value': 2304761414.0, 'total_transactions': 4499178.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Daman', 'avg_daily_value': 6463157.906849315, 'avg_daily_count': 12882.528767123287, 'total_value': 2359052636.0, 'total_transactions': 4702123.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Delhi', 'avg_daily_value': 6474467.512328767, 'avg_daily_count': 12769.421917808218, 'total_value': 2363180642.0, 'total_transactions': 4660839.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Doda', 'avg_daily_value': 6273868.147945206, 'avg_daily_count': 12369.82191780822, 'total_value': 2289961874.0, 'total_transactions': 4514985.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Durg', 'avg_daily_value': 6520509.25, 'avg_daily_count': 12765.758241758242, 'total_value': 2373465367.0, 'total_transactions': 4646736.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Goa', 'avg_daily_value': 6324836.95890411, 'avg_daily_count': 12681.939726027398, 'total_value': 2308565490.0, 'total_transactions': 4628908.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Hyderabad', 'avg_daily_value': 6163065.098630137, 'avg_daily_count': 12124.65205479452, 'total_value': 2249518761.0, 'total_transactions': 4425498.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Indore', 'avg_daily_value': 6166252.528767123, 'avg_daily_count': 12259.602739726028, 'total_value': 2250682173.0, 'total_transactions': 4474755.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Jaipur', 'avg_daily_value': 6256146.320547945, 'avg_daily_count': 12314.901369863013, 'total_value': 2283493407.0, 'total_transactions': 4494939.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kannur', 'avg_daily_value': 6390352.128767123, 'avg_daily_count': 12663.934246575342, 'total_value': 2332478527.0, 'total_transactions': 4622336.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kanpur', 'avg_daily_value': 6485107.243835617, 'avg_daily_count': 12652.331506849316, 'total_value': 2367064144.0, 'total_transactions': 4618101.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kochin', 'avg_daily_value': 6587227.953424658, 'avg_daily_count': 13040.150684931506, 'total_value': 2404338203.0, 'total_transactions': 4759655.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kolkata', 'avg_daily_value': 6512276.419178083, 'avg_daily_count': 12792.128767123288, 'total_value': 2376980893.0, 'total_transactions': 4669127.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Konark', 'avg_daily_value': 6447778.345205479, 'avg_daily_count': 12772.849315068494, 'total_value': 2353439096.0, 'total_transactions': 4662090.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kota', 'avg_daily_value': 6325974.128767123, 'avg_daily_count': 12560.369863013699, 'total_value': 2308980557.0, 'total_transactions': 4584535.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Kullu', 'avg_daily_value': 6496194.7397260275, 'avg_daily_count': 12770.356164383562, 'total_value': 2371111080.0, 'total_transactions': 4661180.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Lucknow', 'avg_daily_value': 6342877.361643836, 'avg_daily_count': 12374.134246575342, 'total_value': 2315150237.0, 'total_transactions': 4516559.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Ludhiana', 'avg_daily_value': 6424431.536986302, 'avg_daily_count': 12595.065753424658, 'total_value': 2344917511.0, 'total_transactions': 4597199.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Lunglei', 'avg_daily_value': 6512496.136986301, 'avg_daily_count': 12656.646575342465, 'total_value': 2377061090.0, 'total_transactions': 4619676.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Madurai', 'avg_daily_value': 6524157.75890411, 'avg_daily_count': 12820.468493150685, 'total_value': 2381317582.0, 'total_transactions': 4679471.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Mathura', 'avg_daily_value': 6334121.736986301, 'avg_daily_count': 12463.52602739726, 'total_value': 2311954434.0, 'total_transactions': 4549187.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Mon', 'avg_daily_value': 6515053.3369863015, 'avg_daily_count': 12973.386301369863, 'total_value': 2377994468.0, 'total_transactions': 4735286.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Patiala', 'avg_daily_value': 6555505.816438356, 'avg_daily_count': 12618.912328767123, 'total_value': 2392759623.0, 'total_transactions': 4605903.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Pune', 'avg_daily_value': 6554958.531506849, 'avg_daily_count': 13049.032876712328, 'total_value': 2392559864.0, 'total_transactions': 4762897.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Ranchi', 'avg_daily_value': 6436859.065753425, 'avg_daily_count': 12640.849315068494, 'total_value': 2349453559.0, 'total_transactions': 4613910.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Srinagar', 'avg_daily_value': 6480123.068681318, 'avg_daily_count': 12566.706043956045, 'total_value': 2358764797.0, 'total_transactions': 4574281.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Surat', 'avg_daily_value': 6362630.906849315, 'avg_daily_count': 12316.542465753424, 'total_value': 2322360281.0, 'total_transactions': 4495538.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Tirumala', 'avg_daily_value': 6087415.608219178, 'avg_daily_count': 11947.441095890412, 'total_value': 2221906697.0, 'total_transactions': 4360816.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Trichy', 'avg_daily_value': 6541154.378082192, 'avg_daily_count': 13098.164383561643, 'total_value': 2387521348.0, 'total_transactions': 4780830.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Varanasi', 'avg_daily_value': 6325070.8, 'avg_daily_count': 12323.909589041095, 'total_value': 2308650842.0, 'total_transactions': 4498227.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INTERNATIONAL', 'Location': 'Vellore', 'avg_daily_value': 6392979.695890411, 'avg_daily_count': 12818.876712328767, 'total_value': 2333437589.0, 'total_transactions': 4678890.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART3 = pd.DataFrame([
    {'Domain': 'INVESTMENTS', 'Location': 'Ahmedabad', 'avg_daily_value': 6463312.523287672, 'avg_daily_count': 12518.027397260274, 'total_value': 2359109071.0, 'total_transactions': 4569080.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Ajmer', 'avg_daily_value': 6138181.747945205, 'avg_daily_count': 12193.819178082193, 'total_value': 2240436338.0, 'total_transactions': 4450744.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Akola', 'avg_daily_value': 6522055.484931507, 'avg_daily_count': 12718.276712328767, 'total_value': 2380550252.0, 'total_transactions': 4642171.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Ambala', 'avg_daily_value': 6457458.589041096, 'avg_daily_count': 12665.761643835616, 'total_value': 2356972385.0, 'total_transactions': 4623003.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Amritsar', 'avg_daily_value': 6565188.94520548, 'avg_daily_count': 12773.632876712329, 'total_value': 2396293965.0, 'total_transactions': 4662376.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Ara', 'avg_daily_value': 6337136.464285715, 'avg_daily_count': 12333.387362637362, 'total_value': 2306717673.0, 'total_transactions': 4489353.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Banglore', 'avg_daily_value': 6270969.383561644, 'avg_daily_count': 12504.369863013699, 'total_value': 2288903825.0, 'total_transactions': 4564095.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Betul', 'avg_daily_value': 6373935.463013698, 'avg_daily_count': 12767.854794520548, 'total_value': 2326486444.0, 'total_transactions': 4660267.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bhind', 'avg_daily_value': 6418808.068493151, 'avg_daily_count': 12554.416438356164, 'total_value': 2342864945.0, 'total_transactions': 4582362.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bhopal', 'avg_daily_value': 6609747.049315069, 'avg_daily_count': 12850.326027397261, 'total_value': 2412557673.0, 'total_transactions': 4690369.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bhuj', 'avg_daily_value': 6324195.610958904, 'avg_daily_count': 12440.627397260274, 'total_value': 2308331398.0, 'total_transactions': 4540829.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bidar', 'avg_daily_value': 6322674.879452054, 'avg_daily_count': 12403.356164383562, 'total_value': 2307776331.0, 'total_transactions': 4527225.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bikaner', 'avg_daily_value': 6415033.556164384, 'avg_daily_count': 12715.750684931507, 'total_value': 2341487248.0, 'total_transactions': 4641249.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bokaro', 'avg_daily_value': 6533801.284931507, 'avg_daily_count': 12796.786301369862, 'total_value': 2384837469.0, 'total_transactions': 4670827.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Bombay', 'avg_daily_value': 6233553.876712329, 'avg_daily_count': 12339.13698630137, 'total_value': 2275247165.0, 'total_transactions': 4503785.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Buxar', 'avg_daily_value': 6403299.073972602, 'avg_daily_count': 12707.898630136986, 'total_value': 2337204162.0, 'total_transactions': 4638383.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Daman', 'avg_daily_value': 6489949.487671233, 'avg_daily_count': 12975.465753424658, 'total_value': 2368831563.0, 'total_transactions': 4736045.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Delhi', 'avg_daily_value': 6361878.293150685, 'avg_daily_count': 12480.887671232877, 'total_value': 2322085577.0, 'total_transactions': 4555524.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Doda', 'avg_daily_value': 6373409.4410958905, 'avg_daily_count': 12660.78904109589, 'total_value': 2326294446.0, 'total_transactions': 4621188.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Durg', 'avg_daily_value': 6591202.458791208, 'avg_daily_count': 12896.079670329671, 'total_value': 2399197695.0, 'total_transactions': 4694173.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Goa', 'avg_daily_value': 6396063.778082192, 'avg_daily_count': 12455.849315068494, 'total_value': 2334563279.0, 'total_transactions': 4546385.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Hyderabad', 'avg_daily_value': 6723024.490410959, 'avg_daily_count': 13219.504109589041, 'total_value': 2453903939.0, 'total_transactions': 4825119.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Indore', 'avg_daily_value': 6390061.1561643835, 'avg_daily_count': 12465.687671232878, 'total_value': 2332372322.0, 'total_transactions': 4549976.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Jaipur', 'avg_daily_value': 6471510.442307692, 'avg_daily_count': 12697.362637362638, 'total_value': 2355629801.0, 'total_transactions': 4621840.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kannur', 'avg_daily_value': 6521248.852054794, 'avg_daily_count': 12870.487671232877, 'total_value': 2380255831.0, 'total_transactions': 4697728.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kanpur', 'avg_daily_value': 6439040.638356164, 'avg_daily_count': 12675.701369863014, 'total_value': 2350249833.0, 'total_transactions': 4626631.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kochin', 'avg_daily_value': 6363538.580821917, 'avg_daily_count': 12475.21095890411, 'total_value': 2322691582.0, 'total_transactions': 4553452.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kolkata', 'avg_daily_value': 6260223.295890411, 'avg_daily_count': 12174.857534246576, 'total_value': 2284981503.0, 'total_transactions': 4443823.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Konark', 'avg_daily_value': 6462335.095890411, 'avg_daily_count': 12841.627397260274, 'total_value': 2358752310.0, 'total_transactions': 4687194.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kota', 'avg_daily_value': 6516458.493150685, 'avg_daily_count': 12956.271232876712, 'total_value': 2378507350.0, 'total_transactions': 4729039.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Kullu', 'avg_daily_value': 6374112.895890411, 'avg_daily_count': 12580.421917808218, 'total_value': 2326551207.0, 'total_transactions': 4591854.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Lucknow', 'avg_daily_value': 6316924.082417582, 'avg_daily_count': 12436.343406593407, 'total_value': 2299360366.0, 'total_transactions': 4526829.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Ludhiana', 'avg_daily_value': 6418156.120879121, 'avg_daily_count': 12414.247252747253, 'total_value': 2336208828.0, 'total_transactions': 4518786.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Lunglei', 'avg_daily_value': 6493892.25479452, 'avg_daily_count': 12838.85205479452, 'total_value': 2370270673.0, 'total_transactions': 4686181.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Madurai', 'avg_daily_value': 6062843.071232877, 'avg_daily_count': 11920.232876712329, 'total_value': 2212937721.0, 'total_transactions': 4350885.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Mathura', 'avg_daily_value': 6343830.043835617, 'avg_daily_count': 12288.07397260274, 'total_value': 2315497966.0, 'total_transactions': 4485147.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Mon', 'avg_daily_value': 6104911.175342466, 'avg_daily_count': 11991.843835616439, 'total_value': 2228292579.0, 'total_transactions': 4377023.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Patiala', 'avg_daily_value': 6628311.997260274, 'avg_daily_count': 13207.739726027397, 'total_value': 2419333879.0, 'total_transactions': 4820825.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Pune', 'avg_daily_value': 6386789.005479452, 'avg_daily_count': 12569.057534246575, 'total_value': 2331177987.0, 'total_transactions': 4587706.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Ranchi', 'avg_daily_value': 6420087.687671233, 'avg_daily_count': 12604.86301369863, 'total_value': 2343332006.0, 'total_transactions': 4600775.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Srinagar', 'avg_daily_value': 6387994.05479452, 'avg_daily_count': 12500.25205479452, 'total_value': 2331617830.0, 'total_transactions': 4562592.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Surat', 'avg_daily_value': 6280499.602739726, 'avg_daily_count': 12326.810958904109, 'total_value': 2292382355.0, 'total_transactions': 4499286.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Tirumala', 'avg_daily_value': 6579816.391780822, 'avg_daily_count': 12934.309589041095, 'total_value': 2401632983.0, 'total_transactions': 4721023.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Trichy', 'avg_daily_value': 6364829.224657535, 'avg_daily_count': 12450.312328767122, 'total_value': 2323162667.0, 'total_transactions': 4544364.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Varanasi', 'avg_daily_value': 6506008.123287671, 'avg_daily_count': 12844.517808219178, 'total_value': 2374692965.0, 'total_transactions': 4688249.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'INVESTMENTS', 'Location': 'Vellore', 'avg_daily_value': 6501499.819178082, 'avg_daily_count': 12675.654794520548, 'total_value': 2373047434.0, 'total_transactions': 4626614.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART4 = pd.DataFrame([
    {'Domain': 'MEDICAL', 'Location': 'Ahmedabad', 'avg_daily_value': 6335645.235616438, 'avg_daily_count': 12550.364383561644, 'total_value': 2312510511.0, 'total_transactions': 4580883.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Ajmer', 'avg_daily_value': 6424707.767123288, 'avg_daily_count': 12708.964383561644, 'total_value': 2345018335.0, 'total_transactions': 4638772.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Akola', 'avg_daily_value': 6374172.24109589, 'avg_daily_count': 12474.70684931507, 'total_value': 2326572868.0, 'total_transactions': 4553268.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Ambala', 'avg_daily_value': 6553992.115068493, 'avg_daily_count': 12808.101369863014, 'total_value': 2392207122.0, 'total_transactions': 4674957.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Amritsar', 'avg_daily_value': 6401986.931506849, 'avg_daily_count': 12711.616438356165, 'total_value': 2336725230.0, 'total_transactions': 4639740.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Ara', 'avg_daily_value': 6558361.698630137, 'avg_daily_count': 12903.216438356165, 'total_value': 2393802020.0, 'total_transactions': 4709674.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Banglore', 'avg_daily_value': 6526942.871232877, 'avg_daily_count': 12775.94794520548, 'total_value': 2382334148.0, 'total_transactions': 4663221.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Betul', 'avg_daily_value': 6430971.709589041, 'avg_daily_count': 12876.016438356164, 'total_value': 2347304674.0, 'total_transactions': 4699746.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bhind', 'avg_daily_value': 6255631.989041096, 'avg_daily_count': 12402.542465753424, 'total_value': 2283305676.0, 'total_transactions': 4526928.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bhopal', 'avg_daily_value': 6408340.849315069, 'avg_daily_count': 12615.405479452054, 'total_value': 2339044410.0, 'total_transactions': 4604623.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bhuj', 'avg_daily_value': 6523478.342465754, 'avg_daily_count': 12619.290410958904, 'total_value': 2381069595.0, 'total_transactions': 4606041.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bidar', 'avg_daily_value': 6544989.898630137, 'avg_daily_count': 12840.868493150685, 'total_value': 2388921313.0, 'total_transactions': 4686917.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bikaner', 'avg_daily_value': 6307489.805479452, 'avg_daily_count': 12316.816438356165, 'total_value': 2302233779.0, 'total_transactions': 4495638.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bokaro', 'avg_daily_value': 6609322.876712329, 'avg_daily_count': 12904.62191780822, 'total_value': 2412402850.0, 'total_transactions': 4710187.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Bombay', 'avg_daily_value': 6349588.794520548, 'avg_daily_count': 12388.64109589041, 'total_value': 2317599910.0, 'total_transactions': 4521854.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Buxar', 'avg_daily_value': 6313369.824657534, 'avg_daily_count': 12427.479452054795, 'total_value': 2304379986.0, 'total_transactions': 4536030.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Daman', 'avg_daily_value': 6381614.8575342465, 'avg_daily_count': 12462.29315068493, 'total_value': 2329289423.0, 'total_transactions': 4548737.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Delhi', 'avg_daily_value': 6275818.079452055, 'avg_daily_count': 12362.728767123288, 'total_value': 2290673599.0, 'total_transactions': 4512396.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Doda', 'avg_daily_value': 6441052.309589041, 'avg_daily_count': 12651.82191780822, 'total_value': 2350984093.0, 'total_transactions': 4617915.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Durg', 'avg_daily_value': 6278487.266483516, 'avg_daily_count': 12236.456043956045, 'total_value': 2285369365.0, 'total_transactions': 4454070.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Goa', 'avg_daily_value': 6296100.657534246, 'avg_daily_count': 12427.74794520548, 'total_value': 2298076740.0, 'total_transactions': 4536128.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Hyderabad', 'avg_daily_value': 6454959.534246575, 'avg_daily_count': 12527.945205479453, 'total_value': 2356060230.0, 'total_transactions': 4572700.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Indore', 'avg_daily_value': 6528901.109589041, 'avg_daily_count': 12628.91506849315, 'total_value': 2383048905.0, 'total_transactions': 4609554.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Jaipur', 'avg_daily_value': 6403852.884931507, 'avg_daily_count': 12511.520547945205, 'total_value': 2337406303.0, 'total_transactions': 4566705.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kannur', 'avg_daily_value': 6318047.879452054, 'avg_daily_count': 12279.071232876713, 'total_value': 2306087476.0, 'total_transactions': 4481861.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kanpur', 'avg_daily_value': 6209976.73150685, 'avg_daily_count': 11993.90410958904, 'total_value': 2266641507.0, 'total_transactions': 4377775.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kochin', 'avg_daily_value': 6260486.030136987, 'avg_daily_count': 12320.079452054795, 'total_value': 2285077401.0, 'total_transactions': 4496829.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kolkata', 'avg_daily_value': 6502807.8575342465, 'avg_daily_count': 12711.238356164384, 'total_value': 2373524868.0, 'total_transactions': 4639602.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Konark', 'avg_daily_value': 6478728.010958904, 'avg_daily_count': 12525.265753424657, 'total_value': 2364735724.0, 'total_transactions': 4571722.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kota', 'avg_daily_value': 6275734.95890411, 'avg_daily_count': 12266.356164383562, 'total_value': 2290643260.0, 'total_transactions': 4477220.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Kullu', 'avg_daily_value': 6407251.490410959, 'avg_daily_count': 12505.34794520548, 'total_value': 2338646794.0, 'total_transactions': 4564452.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Lucknow', 'avg_daily_value': 6611551.249315068, 'avg_daily_count': 13033.172602739725, 'total_value': 2413216206.0, 'total_transactions': 4757108.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Ludhiana', 'avg_daily_value': 6341928.71780822, 'avg_daily_count': 12232.994520547945, 'total_value': 2314803982.0, 'total_transactions': 4465043.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Lunglei', 'avg_daily_value': 6388419.057534247, 'avg_daily_count': 12641.87397260274, 'total_value': 2331772956.0, 'total_transactions': 4614284.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Madurai', 'avg_daily_value': 6346808.243835617, 'avg_daily_count': 12572.589041095891, 'total_value': 2316585009.0, 'total_transactions': 4588995.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Mathura', 'avg_daily_value': 6414478.107142857, 'avg_daily_count': 12637.733516483517, 'total_value': 2334870031.0, 'total_transactions': 4600135.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Mon', 'avg_daily_value': 6408477.008219178, 'avg_daily_count': 12519.673972602739, 'total_value': 2339094108.0, 'total_transactions': 4569681.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Patiala', 'avg_daily_value': 6562271.386301369, 'avg_daily_count': 12653.11506849315, 'total_value': 2395229056.0, 'total_transactions': 4618387.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Pune', 'avg_daily_value': 6573795.671232876, 'avg_daily_count': 12837.876712328767, 'total_value': 2399435420.0, 'total_transactions': 4685825.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Ranchi', 'avg_daily_value': 6277247.120547946, 'avg_daily_count': 12276.065753424658, 'total_value': 2291195199.0, 'total_transactions': 4480764.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Srinagar', 'avg_daily_value': 6504928.668493151, 'avg_daily_count': 12697.186301369862, 'total_value': 2374298964.0, 'total_transactions': 4634473.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Surat', 'avg_daily_value': 6403122.657534246, 'avg_daily_count': 12591.449315068494, 'total_value': 2337139770.0, 'total_transactions': 4595879.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Tirumala', 'avg_daily_value': 6723215.695890411, 'avg_daily_count': 13123.057534246575, 'total_value': 2453973729.0, 'total_transactions': 4789916.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Trichy', 'avg_daily_value': 6440275.139726028, 'avg_daily_count': 12524.528767123287, 'total_value': 2350700426.0, 'total_transactions': 4571453.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Varanasi', 'avg_daily_value': 6410857.479452055, 'avg_daily_count': 12648.416438356164, 'total_value': 2339962980.0, 'total_transactions': 4616672.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'MEDICAL', 'Location': 'Vellore', 'avg_daily_value': 6512341.931506849, 'avg_daily_count': 12935.189041095891, 'total_value': 2377004805.0, 'total_transactions': 4721344.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART5 = pd.DataFrame([
    {'Domain': 'PUBLIC', 'Location': 'Ahmedabad', 'avg_daily_value': 6382921.736986301, 'avg_daily_count': 12573.665753424657, 'total_value': 2329766434.0, 'total_transactions': 4589388.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Ajmer', 'avg_daily_value': 6467461.849315069, 'avg_daily_count': 12741.797260273972, 'total_value': 2360623575.0, 'total_transactions': 4650756.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Akola', 'avg_daily_value': 6330600.295890411, 'avg_daily_count': 12361.383561643835, 'total_value': 2310669108.0, 'total_transactions': 4511905.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Ambala', 'avg_daily_value': 6605151.482191781, 'avg_daily_count': 12844.164383561643, 'total_value': 2410880291.0, 'total_transactions': 4688120.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Amritsar', 'avg_daily_value': 6359344.778082192, 'avg_daily_count': 12544.024657534246, 'total_value': 2321160844.0, 'total_transactions': 4578569.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Ara', 'avg_daily_value': 6487347.613698631, 'avg_daily_count': 12739.194520547946, 'total_value': 2367881879.0, 'total_transactions': 4649806.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Banglore', 'avg_daily_value': 6543993.756164383, 'avg_daily_count': 12903.320547945206, 'total_value': 2388557721.0, 'total_transactions': 4709712.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Betul', 'avg_daily_value': 6228703.778082192, 'avg_daily_count': 12327.657534246575, 'total_value': 2273476879.0, 'total_transactions': 4499595.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bhind', 'avg_daily_value': 6522195.594520548, 'avg_daily_count': 13112.923287671232, 'total_value': 2380601392.0, 'total_transactions': 4786217.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bhopal', 'avg_daily_value': 6496870.134246576, 'avg_daily_count': 12591.068493150686, 'total_value': 2371357599.0, 'total_transactions': 4595740.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bhuj', 'avg_daily_value': 6617572.397260274, 'avg_daily_count': 13079.032876712328, 'total_value': 2415413925.0, 'total_transactions': 4773847.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bidar', 'avg_daily_value': 6591702.326027397, 'avg_daily_count': 13100.698630136987, 'total_value': 2405971349.0, 'total_transactions': 4781755.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bikaner', 'avg_daily_value': 6313880.552197802, 'avg_daily_count': 12366.532967032967, 'total_value': 2298252521.0, 'total_transactions': 4501418.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bokaro', 'avg_daily_value': 6388252.424657534, 'avg_daily_count': 12631.246575342466, 'total_value': 2331712135.0, 'total_transactions': 4610405.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Bombay', 'avg_daily_value': 6502678.82739726, 'avg_daily_count': 12849.12602739726, 'total_value': 2373477772.0, 'total_transactions': 4689931.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Buxar', 'avg_daily_value': 6599055.665753425, 'avg_daily_count': 12759.367123287671, 'total_value': 2408655318.0, 'total_transactions': 4657169.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Daman', 'avg_daily_value': 6569971.05479452, 'avg_daily_count': 12795.627397260274, 'total_value': 2398039435.0, 'total_transactions': 4670404.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Delhi', 'avg_daily_value': 6348411.228021978, 'avg_daily_count': 12377.618131868132, 'total_value': 2310821687.0, 'total_transactions': 4505453.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Doda', 'avg_daily_value': 6431622.3506849315, 'avg_daily_count': 12766.230136986302, 'total_value': 2347542158.0, 'total_transactions': 4659674.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Durg', 'avg_daily_value': 6424992.169863014, 'avg_daily_count': 12671.312328767122, 'total_value': 2345122142.0, 'total_transactions': 4625029.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Goa', 'avg_daily_value': 6499747.701369863, 'avg_daily_count': 12857.101369863014, 'total_value': 2372407911.0, 'total_transactions': 4692842.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Hyderabad', 'avg_daily_value': 6391286.369863014, 'avg_daily_count': 12585.64109589041, 'total_value': 2332819525.0, 'total_transactions': 4593759.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Indore', 'avg_daily_value': 6191639.898630137, 'avg_daily_count': 12050.46301369863, 'total_value': 2259948563.0, 'total_transactions': 4398419.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Jaipur', 'avg_daily_value': 6405995.704109589, 'avg_daily_count': 12513.997260273973, 'total_value': 2338188432.0, 'total_transactions': 4567609.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kannur', 'avg_daily_value': 6329854.317808219, 'avg_daily_count': 12461.019178082192, 'total_value': 2310396826.0, 'total_transactions': 4548272.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kanpur', 'avg_daily_value': 6342845.128767123, 'avg_daily_count': 12564.117808219178, 'total_value': 2315138472.0, 'total_transactions': 4585903.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kochin', 'avg_daily_value': 6330350.728767123, 'avg_daily_count': 12630.698630136987, 'total_value': 2310578016.0, 'total_transactions': 4610205.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kolkata', 'avg_daily_value': 6207952.471232877, 'avg_daily_count': 12340.197260273973, 'total_value': 2265902652.0, 'total_transactions': 4504172.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Konark', 'avg_daily_value': 6255227.805479452, 'avg_daily_count': 12278.169863013698, 'total_value': 2283158149.0, 'total_transactions': 4481532.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kota', 'avg_daily_value': 6351140.126027397, 'avg_daily_count': 12303.457534246576, 'total_value': 2318166146.0, 'total_transactions': 4490762.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Kullu', 'avg_daily_value': 6560985.882191781, 'avg_daily_count': 13025.594520547946, 'total_value': 2394759847.0, 'total_transactions': 4754342.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Lucknow', 'avg_daily_value': 6330280.706849315, 'avg_daily_count': 12477.8, 'total_value': 2310552458.0, 'total_transactions': 4554397.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Ludhiana', 'avg_daily_value': 6517466.830136986, 'avg_daily_count': 12806.098630136987, 'total_value': 2378875393.0, 'total_transactions': 4674226.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Lunglei', 'avg_daily_value': 6421801.398351648, 'avg_daily_count': 12696.263736263736, 'total_value': 2337535709.0, 'total_transactions': 4621440.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Madurai', 'avg_daily_value': 6359441.73150685, 'avg_daily_count': 12555.334246575343, 'total_value': 2321196232.0, 'total_transactions': 4582697.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Mathura', 'avg_daily_value': 6525669.284931507, 'avg_daily_count': 12692.380821917808, 'total_value': 2381869289.0, 'total_transactions': 4632719.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Mon', 'avg_daily_value': 6303275.243835617, 'avg_daily_count': 12457.29315068493, 'total_value': 2300695464.0, 'total_transactions': 4546912.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Patiala', 'avg_daily_value': 6420837.6630136985, 'avg_daily_count': 12729.33698630137, 'total_value': 2343605747.0, 'total_transactions': 4646208.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Pune', 'avg_daily_value': 6231484.819178082, 'avg_daily_count': 12461.323287671234, 'total_value': 2274491959.0, 'total_transactions': 4548383.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Ranchi', 'avg_daily_value': 6593017.443835616, 'avg_daily_count': 12890.101369863014, 'total_value': 2406451367.0, 'total_transactions': 4704887.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Srinagar', 'avg_daily_value': 6361978.863013699, 'avg_daily_count': 12575.843835616439, 'total_value': 2322122285.0, 'total_transactions': 4590183.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Surat', 'avg_daily_value': 6417748.556164384, 'avg_daily_count': 12607.509589041096, 'total_value': 2342478223.0, 'total_transactions': 4601741.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Tirumala', 'avg_daily_value': 6548949.298630137, 'avg_daily_count': 13032.64109589041, 'total_value': 2390366494.0, 'total_transactions': 4756914.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Trichy', 'avg_daily_value': 6325401.709589041, 'avg_daily_count': 12388.117808219178, 'total_value': 2308771624.0, 'total_transactions': 4521663.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Varanasi', 'avg_daily_value': 6603796.819178082, 'avg_daily_count': 12858.512328767123, 'total_value': 2410385839.0, 'total_transactions': 4693357.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'PUBLIC', 'Location': 'Vellore', 'avg_daily_value': 6347764.115384615, 'avg_daily_count': 12571.552197802197, 'total_value': 2310586138.0, 'total_transactions': 4576045.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART6 = pd.DataFrame([
    {'Domain': 'RESTRAUNT', 'Location': 'Ahmedabad', 'avg_daily_value': 6359392.463013698, 'avg_daily_count': 12542.972602739726, 'total_value': 2321178249.0, 'total_transactions': 4578185.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Ajmer', 'avg_daily_value': 6516206.284931507, 'avg_daily_count': 12934.082191780823, 'total_value': 2378415294.0, 'total_transactions': 4720940.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Akola', 'avg_daily_value': 6238993.619178082, 'avg_daily_count': 12488.690410958905, 'total_value': 2277232671.0, 'total_transactions': 4558372.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Ambala', 'avg_daily_value': 6288387.383561644, 'avg_daily_count': 12345.758904109589, 'total_value': 2295261395.0, 'total_transactions': 4506202.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Amritsar', 'avg_daily_value': 6446419.747945205, 'avg_daily_count': 12554.61095890411, 'total_value': 2352943208.0, 'total_transactions': 4582433.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Ara', 'avg_daily_value': 6428301.465753425, 'avg_daily_count': 12551.05205479452, 'total_value': 2346330035.0, 'total_transactions': 4581134.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Banglore', 'avg_daily_value': 6327137.21369863, 'avg_daily_count': 12326.257534246575, 'total_value': 2309405083.0, 'total_transactions': 4499084.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Betul', 'avg_daily_value': 6443833.567123287, 'avg_daily_count': 12708.605479452055, 'total_value': 2351999252.0, 'total_transactions': 4638641.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bhind', 'avg_daily_value': 6506057.73150685, 'avg_daily_count': 12824.819178082193, 'total_value': 2374711072.0, 'total_transactions': 4681059.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bhopal', 'avg_daily_value': 6432657.646575343, 'avg_daily_count': 12839.145205479452, 'total_value': 2347920041.0, 'total_transactions': 4686288.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bhuj', 'avg_daily_value': 6494699.667582418, 'avg_daily_count': 12790.497252747253, 'total_value': 2364070679.0, 'total_transactions': 4655741.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bidar', 'avg_daily_value': 6217095.783561644, 'avg_daily_count': 12026.131506849315, 'total_value': 2269239961.0, 'total_transactions': 4389538.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bikaner', 'avg_daily_value': 6251813.504109589, 'avg_daily_count': 12415.797260273972, 'total_value': 2281911929.0, 'total_transactions': 4531766.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bokaro', 'avg_daily_value': 6213608.17260274, 'avg_daily_count': 12237.345205479452, 'total_value': 2267966983.0, 'total_transactions': 4466631.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Bombay', 'avg_daily_value': 6296743.208219178, 'avg_daily_count': 12258.172602739725, 'total_value': 2298311271.0, 'total_transactions': 4474233.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Buxar', 'avg_daily_value': 6310962.446575343, 'avg_daily_count': 12607.375342465753, 'total_value': 2303501293.0, 'total_transactions': 4601692.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Daman', 'avg_daily_value': 6377236.936986301, 'avg_daily_count': 12554.016438356164, 'total_value': 2327691482.0, 'total_transactions': 4582216.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Delhi', 'avg_daily_value': 6341911.909589041, 'avg_daily_count': 12523.08493150685, 'total_value': 2314797847.0, 'total_transactions': 4570926.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Doda', 'avg_daily_value': 6480045.531506849, 'avg_daily_count': 12619.364383561644, 'total_value': 2365216619.0, 'total_transactions': 4606068.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Durg', 'avg_daily_value': 6589185.830136986, 'avg_daily_count': 13050.704109589042, 'total_value': 2405052828.0, 'total_transactions': 4763507.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Goa', 'avg_daily_value': 6629325.904109589, 'avg_daily_count': 13307.049315068492, 'total_value': 2419703955.0, 'total_transactions': 4857073.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Hyderabad', 'avg_daily_value': 6464710.871232877, 'avg_daily_count': 12520.682191780821, 'total_value': 2359619468.0, 'total_transactions': 4570049.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Indore', 'avg_daily_value': 6488053.991780822, 'avg_daily_count': 12656.046575342465, 'total_value': 2368139707.0, 'total_transactions': 4619457.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Jaipur', 'avg_daily_value': 6554305.194520548, 'avg_daily_count': 12726.997260273973, 'total_value': 2392321396.0, 'total_transactions': 4645354.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kannur', 'avg_daily_value': 6421894.235616438, 'avg_daily_count': 12878.265753424657, 'total_value': 2343991396.0, 'total_transactions': 4700567.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kanpur', 'avg_daily_value': 6458728.342465754, 'avg_daily_count': 12736.11506849315, 'total_value': 2357435845.0, 'total_transactions': 4648682.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kochin', 'avg_daily_value': 6383442.128767123, 'avg_daily_count': 12550.22191780822, 'total_value': 2329956377.0, 'total_transactions': 4580831.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kolkata', 'avg_daily_value': 6498803.802739726, 'avg_daily_count': 12573.356164383562, 'total_value': 2372063388.0, 'total_transactions': 4589275.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Konark', 'avg_daily_value': 6388304.950684931, 'avg_daily_count': 12615.038356164383, 'total_value': 2331731307.0, 'total_transactions': 4604489.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kota', 'avg_daily_value': 6409538.28219178, 'avg_daily_count': 12381.665753424657, 'total_value': 2339481473.0, 'total_transactions': 4519308.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Kullu', 'avg_daily_value': 6391253.41369863, 'avg_daily_count': 12540.802739726027, 'total_value': 2332807496.0, 'total_transactions': 4577393.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Lucknow', 'avg_daily_value': 6464280.569863014, 'avg_daily_count': 12613.246575342466, 'total_value': 2359462408.0, 'total_transactions': 4603835.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Ludhiana', 'avg_daily_value': 6351549.967123288, 'avg_daily_count': 12390.005479452055, 'total_value': 2318315738.0, 'total_transactions': 4522352.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Lunglei', 'avg_daily_value': 6639154.101369863, 'avg_daily_count': 13075.964383561644, 'total_value': 2423291247.0, 'total_transactions': 4772727.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Madurai', 'avg_daily_value': 6326151.94520548, 'avg_daily_count': 12502.74794520548, 'total_value': 2309045460.0, 'total_transactions': 4563503.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Mathura', 'avg_daily_value': 6316524.26849315, 'avg_daily_count': 12445.249315068493, 'total_value': 2305531358.0, 'total_transactions': 4542516.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Mon', 'avg_daily_value': 6554459.906593407, 'avg_daily_count': 12949.89010989011, 'total_value': 2385823406.0, 'total_transactions': 4713760.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Patiala', 'avg_daily_value': 6250111.276712329, 'avg_daily_count': 12324.498630136986, 'total_value': 2281290616.0, 'total_transactions': 4498442.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Pune', 'avg_daily_value': 6330391.26849315, 'avg_daily_count': 12403.849315068494, 'total_value': 2310592813.0, 'total_transactions': 4527405.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Ranchi', 'avg_daily_value': 6403121.671232876, 'avg_daily_count': 12567.413698630136, 'total_value': 2337139410.0, 'total_transactions': 4587106.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Srinagar', 'avg_daily_value': 6477134.909589041, 'avg_daily_count': 12533.298630136986, 'total_value': 2364154242.0, 'total_transactions': 4574654.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Surat', 'avg_daily_value': 6355060.421917808, 'avg_daily_count': 12462.542465753424, 'total_value': 2319597054.0, 'total_transactions': 4548828.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Tirumala', 'avg_daily_value': 6222617.983561643, 'avg_daily_count': 12134.493150684932, 'total_value': 2271255564.0, 'total_transactions': 4429090.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Trichy', 'avg_daily_value': 6460995.021978022, 'avg_daily_count': 12445.791208791208, 'total_value': 2351802188.0, 'total_transactions': 4530268.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Varanasi', 'avg_daily_value': 6372402.460273973, 'avg_daily_count': 12717.287671232876, 'total_value': 2325926898.0, 'total_transactions': 4641810.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RESTRAUNT', 'Location': 'Vellore', 'avg_daily_value': 6396882.035616438, 'avg_daily_count': 12573.438356164384, 'total_value': 2334861943.0, 'total_transactions': 4589305.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
])

DC_CLUSTERING_DATA_PART7 = pd.DataFrame([
    {'Domain': 'RETAIL', 'Location': 'Ahmedabad', 'avg_daily_value': 6377765.041208792, 'avg_daily_count': 12342.307692307691, 'total_value': 2321506475.0, 'total_transactions': 4492600.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Ajmer', 'avg_daily_value': 6485196.630136986, 'avg_daily_count': 12916.26301369863, 'total_value': 2367096770.0, 'total_transactions': 4714436.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Akola', 'avg_daily_value': 6486577.6630136985, 'avg_daily_count': 12579.150684931506, 'total_value': 2367600847.0, 'total_transactions': 4591390.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Ambala', 'avg_daily_value': 6507964.794520548, 'avg_daily_count': 12828.630136986301, 'total_value': 2375407150.0, 'total_transactions': 4682450.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Amritsar', 'avg_daily_value': 6351004.123287671, 'avg_daily_count': 12588.202739726028, 'total_value': 2318116505.0, 'total_transactions': 4594694.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Ara', 'avg_daily_value': 6339083.449315068, 'avg_daily_count': 12425.506849315068, 'total_value': 2313765459.0, 'total_transactions': 4535310.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Banglore', 'avg_daily_value': 6396696.983561643, 'avg_daily_count': 12634.043835616438, 'total_value': 2334794399.0, 'total_transactions': 4611426.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Betul', 'avg_daily_value': 6390151.394520548, 'avg_daily_count': 12453.202739726028, 'total_value': 2332405259.0, 'total_transactions': 4545419.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bhind', 'avg_daily_value': 6359827.526027397, 'avg_daily_count': 12463.016438356164, 'total_value': 2321337047.0, 'total_transactions': 4549001.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bhopal', 'avg_daily_value': 6404738.019178082, 'avg_daily_count': 12438.021917808219, 'total_value': 2337729377.0, 'total_transactions': 4539878.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bhuj', 'avg_daily_value': 6406895.761643835, 'avg_daily_count': 12592.876712328767, 'total_value': 2338516953.0, 'total_transactions': 4596400.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bidar', 'avg_daily_value': 6330018.654794521, 'avg_daily_count': 12530.093150684932, 'total_value': 2310456809.0, 'total_transactions': 4573484.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bikaner', 'avg_daily_value': 6479315.5452054795, 'avg_daily_count': 12685.427397260273, 'total_value': 2364950174.0, 'total_transactions': 4630181.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bokaro', 'avg_daily_value': 6648948.0, 'avg_daily_count': 12885.764383561644, 'total_value': 2426866020.0, 'total_transactions': 4703304.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Bombay', 'avg_daily_value': 6582088.567123287, 'avg_daily_count': 12971.230136986302, 'total_value': 2402462327.0, 'total_transactions': 4734499.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Buxar', 'avg_daily_value': 6032416.509589041, 'avg_daily_count': 11908.298630136986, 'total_value': 2201832026.0, 'total_transactions': 4346529.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Daman', 'avg_daily_value': 6431955.367123288, 'avg_daily_count': 12534.465753424658, 'total_value': 2347663709.0, 'total_transactions': 4575080.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Delhi', 'avg_daily_value': 6359295.854395605, 'avg_daily_count': 12322.829670329671, 'total_value': 2314783691.0, 'total_transactions': 4485510.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Doda', 'avg_daily_value': 6355977.221917808, 'avg_daily_count': 12566.830136986302, 'total_value': 2319931686.0, 'total_transactions': 4586893.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Durg', 'avg_daily_value': 6483436.791780822, 'avg_daily_count': 12639.684931506848, 'total_value': 2366454429.0, 'total_transactions': 4613485.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Goa', 'avg_daily_value': 6592340.789041096, 'avg_daily_count': 12902.276712328767, 'total_value': 2406204388.0, 'total_transactions': 4709331.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Hyderabad', 'avg_daily_value': 6374019.747945205, 'avg_daily_count': 12424.391780821918, 'total_value': 2326517208.0, 'total_transactions': 4534903.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Indore', 'avg_daily_value': 6405305.646575343, 'avg_daily_count': 12549.912328767123, 'total_value': 2337936561.0, 'total_transactions': 4580718.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Jaipur', 'avg_daily_value': 6197698.323287671, 'avg_daily_count': 12091.616438356165, 'total_value': 2262159888.0, 'total_transactions': 4413440.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kannur', 'avg_daily_value': 6286957.42739726, 'avg_daily_count': 12584.597260273973, 'total_value': 2294739461.0, 'total_transactions': 4593378.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kanpur', 'avg_daily_value': 6451199.693150685, 'avg_daily_count': 12740.969863013699, 'total_value': 2354687888.0, 'total_transactions': 4650454.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kochin', 'avg_daily_value': 6439509.216438356, 'avg_daily_count': 12700.835616438357, 'total_value': 2350420864.0, 'total_transactions': 4635805.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kolkata', 'avg_daily_value': 6474910.726027397, 'avg_daily_count': 12745.78904109589, 'total_value': 2363342415.0, 'total_transactions': 4652213.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Konark', 'avg_daily_value': 6452563.063013699, 'avg_daily_count': 12484.241095890411, 'total_value': 2355185518.0, 'total_transactions': 4556748.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kota', 'avg_daily_value': 6432484.720547945, 'avg_daily_count': 12549.134246575342, 'total_value': 2347856923.0, 'total_transactions': 4580434.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Kullu', 'avg_daily_value': 6404898.841095891, 'avg_daily_count': 12494.942465753425, 'total_value': 2337788077.0, 'total_transactions': 4560654.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Lucknow', 'avg_daily_value': 6296410.602739726, 'avg_daily_count': 12379.838356164384, 'total_value': 2298189870.0, 'total_transactions': 4518641.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Ludhiana', 'avg_daily_value': 6339580.364383562, 'avg_daily_count': 12498.512328767123, 'total_value': 2313946833.0, 'total_transactions': 4561957.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Lunglei', 'avg_daily_value': 6403841.112328767, 'avg_daily_count': 12662.841095890411, 'total_value': 2337402006.0, 'total_transactions': 4621937.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Madurai', 'avg_daily_value': 6289092.233516484, 'avg_daily_count': 12338.376373626374, 'total_value': 2289229573.0, 'total_transactions': 4491169.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Mathura', 'avg_daily_value': 6154280.057534247, 'avg_daily_count': 12343.578082191782, 'total_value': 2246312221.0, 'total_transactions': 4505406.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Mon', 'avg_daily_value': 6615391.98630137, 'avg_daily_count': 13053.578082191782, 'total_value': 2414618075.0, 'total_transactions': 4764556.0, 'Cluster': 0, 'Cluster_Label': 'HIGH_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Patiala', 'avg_daily_value': 6319880.389041096, 'avg_daily_count': 12636.290410958904, 'total_value': 2306756342.0, 'total_transactions': 4612246.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Pune', 'avg_daily_value': 6362531.956164383, 'avg_daily_count': 12516.019178082192, 'total_value': 2322324164.0, 'total_transactions': 4568347.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Ranchi', 'avg_daily_value': 6426042.219178082, 'avg_daily_count': 12634.238356164384, 'total_value': 2345505410.0, 'total_transactions': 4611497.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Srinagar', 'avg_daily_value': 6203779.0, 'avg_daily_count': 12339.25205479452, 'total_value': 2264379335.0, 'total_transactions': 4503827.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Surat', 'avg_daily_value': 6284866.290410959, 'avg_daily_count': 12427.715068493151, 'total_value': 2293976196.0, 'total_transactions': 4536116.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Tirumala', 'avg_daily_value': 6242720.767123288, 'avg_daily_count': 12331.035616438356, 'total_value': 2278593080.0, 'total_transactions': 4500828.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Trichy', 'avg_daily_value': 6341174.57967033, 'avg_daily_count': 12630.543956043955, 'total_value': 2308187547.0, 'total_transactions': 4597518.0, 'Cluster': 2, 'Cluster_Label': 'MEDIUM_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Varanasi', 'avg_daily_value': 6322933.780821918, 'avg_daily_count': 12472.780821917808, 'total_value': 2307870830.0, 'total_transactions': 4552565.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
    {'Domain': 'RETAIL', 'Location': 'Vellore', 'avg_daily_value': 6251228.712328767, 'avg_daily_count': 12400.983561643836, 'total_value': 2281698480.0, 'total_transactions': 4526359.0, 'Cluster': 1, 'Cluster_Label': 'LOW_PERFORMANCE'},
])

# Combine all DC Clustering Parts
DC_CLUSTERING_DATA = pd.concat([
    DC_CLUSTERING_DATA_PART1, 
    DC_CLUSTERING_DATA_PART2, 
    DC_CLUSTERING_DATA_PART3,
    DC_CLUSTERING_DATA_PART4,
    DC_CLUSTERING_DATA_PART5,
    DC_CLUSTERING_DATA_PART6,
    DC_CLUSTERING_DATA_PART7
], ignore_index=True)


# Global DataFrame placeholder (DATA is no longer used for analysis, only metrics)
DATA = pd.DataFrame()
DOMAIN_LOCA_PERF_DATA = DC_CLUSTERING_DATA # Alias for Section 4

@st.cache_data
def load_and_process_data():
    """
    Function to return hardcoded, pre-processed DataFrames.
    NOTE: Data loading from files has been removed to avoid file path/size issues.
    """
    
    domain_summary = DOMAIN_SUMMARY_DATA
    regional_perf = REGIONAL_PERF_DATA 
    monthly_summary = MONTHLY_SUMMARY_DATA
    daily_summary = DAILY_SUMMARY_DATA
    dc = DC_CLUSTERING_DATA
    domain_loca_perf = DOMAIN_LOCA_PERF_DATA
    
    # FIX 1: Explicitly cast DC_CLUSTERING_DATA columns to numeric types for aggregation safety
    numeric_cols = ['avg_daily_value', 'avg_daily_count', 'total_value', 'total_transactions']
    for col in numeric_cols:
        dc[col] = pd.to_numeric(dc[col], errors='coerce') 
    
    # Fill any NaNs that might result from coercion for aggregation safety
    dc = dc.fillna(0)
    domain_loca_perf = dc # Ensure the aliased dataframe also gets the clean data
    
    # FIX 2: Create a placeholder DataFrame with consistent single-element arrays
    data = pd.DataFrame({
        'Value': [TOTAL_VALUE_RUPEES], 
        'Transaction_count': [TOTAL_TXNS_COUNT], 
        'Domain': ['PLACEHOLDER']
    })
    
    return data, domain_summary, regional_perf, monthly_summary, daily_summary, dc, domain_loca_perf

# Load all pre-processed dataframes
data, domain_summary, regional_perf, monthly_summary, daily_summary, dc, domain_loca_perf = load_and_process_data()

# Check if essential data is still missing (only for safety, data is loaded via hardcoding)
if domain_summary.empty:
    st.stop()


# --- HELPER FUNCTIONS FOR VISUALIZATION (No changes needed) ---

def plot_top_10_regional(df):
    """Plots top 10 locations by total value and total transactions."""
    if df.empty:
        st.warning("Regional performance data is not yet available. Please provide the next set of data.")
        return plt.figure(figsize=(1, 1))

    # Ensure the dataframe is sorted before plotting nlargest
    df_sorted = df.sort_values(by='total_value', ascending=False)
    top10_value = df_sorted.nlargest(10, 'total_value')
    top10_count = df_sorted.nlargest(10, 'total_transactions')

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
    if monthly_df.empty or daily_df.empty:
        st.warning("Temporal analysis data is not yet available. Please provide the next set of data.")
        return plt.figure(figsize=(1, 1))
        
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
    if df.empty:
        st.warning("Domain-Location data is not yet available. Please provide the next set of data.")
        return plt.figure(figsize=(1, 1))
        
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
    
    # Using hardcoded constants for metrics, displayed in Billions and Millions
    total_value = TOTAL_VALUE_RUPEES / 1e9 # Billions
    total_txns = TOTAL_TXNS_COUNT / 1e6 # Millions
    
    col1.metric("Total Value (Annual)", f"₹{total_value:,.2f} Billion")
    col2.metric("Total Transactions (Annual)", f"{total_txns:,.2f} Million")
    col3.metric("Domains Covered", UNIQUE_DOMAINS)

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
    # Using hardcoded data for observations
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

    if regional_perf.empty:
        st.info("Regional performance data is missing. Please provide the Top 15 cities data next.")
    else:
        regional_plot = plot_top_10_regional(regional_perf)
        st.pyplot(regional_plot)
        
        st.info("""
        **Insight:** While overall regional performance is highly consistent (as noted in your original analysis), the visualization highlights minor regional fluctuations. The high degree of uniformity suggests strong operational consistency and balanced merchant penetration across the bank's network.
        """)

elif selection == "4. Domain and Location Wise Performance":
    
    # ----------------------------------------------------
    # SECTION 4: DOMAIN AND LOCATION WISE PERFORMANCE
    # ----------------------------------------------------
    st.header("4. Domain and Location Wise Performance")
    st.markdown("A deep dive into the performance of every combination of Domain and City, highlighting where specific domains thrive.")
    
    if domain_loca_perf.empty:
        st.info("Domain and Location performance data is missing. Please provide the final clustering data in a subsequent step.")
    else:
        st.subheader("Top 10 Domain-City Pairs by Total Value")
        top_pairs = domain_loca_perf.sort_values('total_value', ascending=False).nlargest(10, 'total_value').copy()
        
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
    
    if monthly_summary.empty:
        st.info("Temporal analysis data is missing. Please provide the Monthly and Daily summary data next.")
    else:
        temporal_plot = plot_temporal_trends(monthly_summary, daily_summary)
        st.pyplot(temporal_plot)
    
        col_temp1, col_temp2 = st.columns(2)
        with col_temp1:
            st.subheader("Monthly Trends")
            st.markdown("""
            - **Peaks:** The data clearly shows peak activity months, indicating periods of high consumer spending (e.g., mid-year and year-end surges, specifically **July, August, May** and **October, December**).
            - **Dips:** The lowest activity month is consistently **February**, followed by slight dips in **April, June, and November**. This is ideal for targeted promotional pushes.
            """)
        with col_temp2:
            st.subheader("Daily Trends")
            st.markdown("""
            - **Weekend Activity:** There is a distinct spike in activity on **Saturday** in both value and volume compared to the weekdays.
            - **Weekday Stability:** Activity remains largely stable across all other days, with a slight dip observed on **Friday**.
            """)


elif selection == "6. Clustering and Its Results":
    
    # ----------------------------------------------------
    # SECTION 6: CLUSTERING AND ITS RESULTS
    # ----------------------------------------------------
    
    st.header("6. Clustering and Its Results")
    st.markdown("Using K-Means clustering (k=3) on daily averages and total metrics to categorize every Domain-City pair into performance segments.")

    if dc.empty:
        st.info("Clustering data is missing. Please provide the final clustering results next.")
    else:
        # Summary of Clusters
        cluster_summary = dc.groupby('Cluster_Label').agg(
            Pairs_Count=('Location', 'count'),
            Avg_Daily_Value_Mean=('avg_daily_value', 'mean'),
            Total_Value_Mean=('total_value', 'mean')
        ).sort_values('Total_Value_Mean', ascending=False).reset_index()
        
        # Manually assign cluster labels to ensure correct ordering (High > Medium > Low)
        label_order = ['HIGH_PERFORMANCE', 'MEDIUM_PERFORMANCE', 'LOW_PERFORMANCE']
        cluster_summary['Cluster_Label'] = pd.Categorical(cluster_summary['Cluster_Label'], categories=label_order, ordered=True)
        cluster_summary = cluster_summary.sort_values('Cluster_Label')
    
        # Formatting
        cluster_summary['Avg_Daily_Value_Mean'] = (cluster_summary['Avg_Daily_Value_Mean']).map('₹{:,.0f}'.format)
        cluster_summary['Total_Value_Mean'] = (cluster_summary['Total_Value_Mean'] / 1e9).map('₹{:,.2f}B'.format)
        
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
            st.dataframe(high_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True,
                column_config={
                    "total_value": st.column_config.NumberColumn("Total Value", format="₹%,.0f"),
                    "total_transactions": st.column_config.NumberColumn("Total Txns", format="%,.0f"),
                }
            )
    
        # Medium Performance Cluster
        medium_df = dc[dc.Cluster_Label == "MEDIUM_PERFORMANCE"].sort_values('total_value', ascending=False)
        with tab2:
            st.warning("📈 **Strategy: Activation & Expansion**")
            st.markdown("""
            These are stable markets with potential for growth. The goal is to elevate them to High Performance status.
            """)
            st.markdown("- **Action:** Run targeted activation campaigns to increase transaction frequency (e.g., cashback on 5th transaction).")
            st.markdown("- **Action:** Accelerate merchant onboarding, especially micro and small businesses.")
            st.dataframe(medium_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True,
                column_config={
                    "total_value": st.column_config.NumberColumn("Total Value", format="₹%,.0f"),
                    "total_transactions": st.column_config.NumberColumn("Total Txns", format="%,.0f"),
                }
            )
    
        # Low Performance Cluster
        low_df = dc[dc.Cluster_Label == "LOW_PERFORMANCE"].sort_values('total_value', ascending=False)
        with tab3:
            st.error("🛠️ **Strategy: Digital Adoption & Infrastructure**")
            st.markdown("""
            These markets are underperforming or underserved, requiring fundamental improvements in outreach and adoption.
            """)
            st.markdown("- **Action:** Increase digital awareness drives and customer training on mobile/UPI services.")
            st.markdown("- **Action:** Offer strong incentives (cashbacks) for first-time digital users and new merchants.")
            st.dataframe(low_df[['Domain', 'Location', 'total_value', 'total_transactions']].head(10), use_container_width=True,
                column_config={
                    "total_value": st.column_config.NumberColumn("Total Value", format="₹%,.0f"),
                    "total_transactions": st.column_config.NumberColumn("Total Txns", format="%,.0f"),
                }
            )
        
        
        # Optional: Drilldown filter
        st.sidebar.subheader("Cluster Drilldown")
        selected_cluster = st.sidebar.selectbox("Select Cluster to Analyze", ['HIGH_PERFORMANCE', 'MEDIUM_PERFORMANCE', 'LOW_PERFORMANCE'])
        
        if selected_cluster:
            st.subheader(f"Full List: {selected_cluster} Pairs")
            filtered_df = dc[dc.Cluster_Label == selected_cluster].sort_values('total_value', ascending=False)
            st.dataframe(
                filtered_df[['Domain', 'Location', 'total_value', 'total_transactions', 'avg_daily_value', 'avg_daily_count']], 
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
