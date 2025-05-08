import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'png'
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.ticker as mtick
import numpy as np
from sklearn.cluster import KMeans
import warnings
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

def ML1_predict(clustering_df, scaled_data):
    kmeans = KMeans(n_clusters=3, random_state=42)
    clustering_df['Cluster'] = kmeans.fit_predict(scaled_data)
    # map cluster labels to affordability level
    cluster_map = {
        0: 'Moderate Affordability',
        1: 'High Affordability',
        2: 'Low Affordability'
    }
    clustering_df['Affordability'] = clustering_df['Cluster'].map(cluster_map)
    n_samples = scaled_data.shape[0]
    np.random.seed(42)
    random_clusters = np.random.randint(0, 3, size=n_samples)
    random_silhouette = silhouette_score(scaled_data, random_clusters)
    kmeans_silhouette = silhouette_score(scaled_data, clustering_df['Cluster'])
    print(f"Silhouette Score (Random Clustering): {random_silhouette:.3f}")
    print(f"Silhouette Score (K-Means Clustering): {kmeans_silhouette:.3f}")
    return clustering_df
    
def cleaning_homevalues(home_values_dataset):
    home_values_dataset.columns.values[5:] = pd.to_datetime(home_values_dataset.columns[5:])
    threshold = 0.4
    home_values = home_values_dataset.loc[home_values_dataset.isnull().mean(axis=1) < threshold]
    # Interpolate across columns (i.e., across time for each region)
    home_values.iloc[:, 5:] = home_values.iloc[:, 5:].interpolate(axis=1)
    # Fill any remaining edge NaNs with forward/backward fill
    home_values.iloc[:, 5:] = home_values.iloc[:, 5:].bfill(axis=1).ffill(axis=1)
    return home_values
    
def homevalues_dataset(home_values):
    # Create an empty list to store each state's yearly average series
    state_rows = []
    
    # Loop over each unique state
    for state in home_values['StateName'].unique():
        # Step 1: Filter data for the state
        state_data = home_values[home_values['StateName'] == state]
        
        # Step 2: Extract and transpose time series
        ts_data = state_data.iloc[:, 5:].T
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data.columns = state_data['RegionID'].values
        
        # Step 3: Group by year and average across regions
        ts_yearly = ts_data.groupby(ts_data.index.year).mean()
        
        # Step 4: Average across all regions in the state per year
        state_avg_series = ts_yearly.mean(axis=1)
        
        # Step 5: Add the state name and store as a dictionary
        state_row = {'StateName': state}
        state_row.update(state_avg_series.to_dict())
        state_rows.append(state_row)
    
    # Create final DataFrame
    state_yearly_df = pd.DataFrame(state_rows)
    
    # Optional: Set StateName as index and sort columns by year
    state_yearly_df = state_yearly_df.set_index('StateName')
    state_yearly_df = state_yearly_df[sorted(state_yearly_df.columns)]
    state_yearly_df = state_yearly_df.drop(index=np.nan)
    return state_yearly_df
    
def income_dataset_ML(income):
    for col in income.columns[1:]:  # skip 'State' column
        income[col] = income[col].replace(',', '', regex=True).astype(float)
    
    # Step 2: Add state codes (e.g. IL, CA, NY)
    # We'll use a mapping from state name to abbreviation
    us_state_abbrev = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
        'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD','Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV','New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND','Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD','Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
        'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV','Wisconsin': 'WI', 'Wyoming': 'WY', 'The United States': 'US'}
    
    # Add a new column with state abbreviations
    income['StateCode'] = income['State'].map(us_state_abbrev)
    
    # Step 3: Optional — ensure year columns are int type (they might be strings)
    income.columns = [int(col) if col.isdigit() else col for col in income.columns]
    income = income[income['State'] != 'The United States']
    return income

def clustering(state_yearly_df, income):
    # Home value current and growth
    home_current = state_yearly_df[2023]
    home_growth = ((state_yearly_df[2023] - state_yearly_df[2000]) / state_yearly_df[2000]) * 100
    
    # Income current and growth
    income_current = income.set_index('StateCode').loc[state_yearly_df.index][2023]
    income_growth = ((income.set_index('StateCode').loc[state_yearly_df.index][2023] -
                      income.set_index('StateCode').loc[state_yearly_df.index][2000]) /
                     income.set_index('StateCode').loc[state_yearly_df.index][2000]) * 100
    
    # Step 2: Create a DataFrame for clustering
    clustering_df = pd.DataFrame({
        'HomeValue_2023': home_current,
        'HomeValue_Growth': home_growth,
        'Income_2023': income_current,
        'Income_Growth': income_growth
    })
    
    # Drop any rows with missing values just in case
    clustering_df = clustering_df.dropna()
    
    return clustering_df

def choropleth_graph(median_income):    
    # Map full state names to abbreviations
    state_to_abbrev = {
        'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA',
        'Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI',
        'Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY',
        'Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI',
        'Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE',
        'Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY',
        'North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR',
        'Pennsylvania': 'PA','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD',
        'Tennessee': 'TN','Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA',
        'Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'
    }

    # Copy the input DataFrame
    df = median_income.copy()

    # Map state names to abbreviations
    df['State_Abbrev'] = df['State'].map(state_to_abbrev)

    # Drop rows where abbreviation mapping failed
    df_clean = df.dropna(subset=['State_Abbrev'])

    # Get year columns (everything except State and Abbrev)
    date_columns = df_clean.columns.difference(['State', 'State_Abbrev'])

    df_clean[date_columns] = df_clean[date_columns].replace(r'[\$,]', '', regex=True).astype(float)

    # Melt the dataframe for animation
    df_long = df_clean.melt(
        id_vars=['State', 'State_Abbrev'],
        value_vars=date_columns,
        var_name='Date',
        value_name='Value'
    )

    # Create choropleth map
    fig = px.choropleth(
        df_long,
        locations='State_Abbrev',
        locationmode='USA-states',
        color='Value',
        scope='usa',
        color_continuous_scale='pinkyl',
        animation_frame='Date',
        title="Average Income by Year"
    )

    # Resize the figure
    fig.update_layout(width=800, height=550)
    fig.show()



def showHeatMap(home_values_dataset,income):
    # Create a copy to avoid changing the original dataset
    home_values_copy = home_values_dataset.copy()
    income_copy = income.copy()
    
    # Manually map state abbreviations to full names
    abbr_to_full = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Map full state names
    home_values_copy['State'] = home_values_copy['StateName'].map(abbr_to_full)
    
    # Convert to long format and extract year
    home_long = home_values_copy.melt(
        id_vars=['State'],
        value_vars=home_values_copy.columns[5:-1],  # exclude trailing columns if any
        var_name='Date',
        value_name='HomeValue'
    )
    home_long['Year'] = pd.to_datetime(home_long['Date']).dt.year
    home_yearly = home_long.groupby(['State', 'Year'])['HomeValue'].mean().reset_index()
    
    # Clean income data (on copy)
    for col in income_copy.columns[1:]:
        income_copy[col] = income_copy[col].str.replace(',', '').astype(float)
    income_long = income_copy.melt(id_vars=['State'], var_name='Year', value_name='MedianIncome')
    income_long['Year'] = income_long['Year'].astype(int)
    
    # Merge and calculate affordability
    merged = pd.merge(income_long, home_yearly, on=['State', 'Year'], how='inner')
    merged['PriceToIncomeRatio'] = merged['HomeValue'] / merged['MedianIncome']
    return merged

def baseline(scaled_data,n1,n2):
    K_range = range(n1, n2)
    
    inertias = []
    silhouette_scores = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)  # scaled_data = standardized feature matrix from earlier
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, labels))
    
    # Plotting
    plt.figure(figsize=(8, 3))
    
    # Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, marker='o')
    plt.title('Elbow Method (Inertia)')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    # Silhouette Score
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, marker='o', color='orange')
    plt.title('Silhouette Score vs K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def showClustersInMap(clustering_df):
    map_df = clustering_df.reset_index()[['Affordability']]
    map_df['StateCode'] = clustering_df.index
    
    fig = px.choropleth(
        map_df,
        locations='StateCode',
        locationmode='USA-states',
        color='Affordability',
        color_discrete_map={
            'High Affordability': '#8CD47E',
            'Moderate Affordability': '#F8D66D',
            'Low Affordability': '#FF6961'
        },
        hover_name='StateCode',
        scope='usa',
        title='Where Can You Still Afford a Home? A Clustering of U.S. States by Affordability'
    )
    
    fig.update_layout(
        geo=dict(bgcolor='rgba(0,0,0,0)'),
        margin={"r":0,"t":40,"l":0,"b":0}
    )
    
    fig.show()

def county_graph():

    sns.set(style="whitegrid")
    
    # Load Data
    income = pd.read_csv('household_income_county.csv')
    income = income[income['Race'] == 'Total']
    income['Ideal_Price'] = income['Household Income by Race'] * 3
    
    county_medians = income.groupby('Geography')['Ideal_Price'].median().reset_index()
    
    baseline_price = 148540.88  # Ideal Housing Price
    
    county_medians = county_medians.sort_values(by='Ideal_Price')
    
    # Set figure size
    plt.figure(figsize=(30,10))
    
    # Plot line graph
    plt.plot(county_medians['Geography'], county_medians['Ideal_Price'], marker='o', label='Median Ideal Housing Price')
    
    # Add baseline housing price line
    plt.axhline(y=baseline_price, color='red', linestyle='--', linewidth=2, label='Avg Housing Price (2023)')
    
    plt.xticks(rotation=60, ha='right')
    
    # Titles and labels
    plt.title('Median Ideal Housing Price vs. Average Housing Price in Illinois Counties (2023)')
    plt.ylabel('Price ($)')
    plt.xlabel('County')
    plt.legend()
    
    plt.tight_layout()
    
    return plt    

def CreateMerged(home_values_dataset,income):
    # Create a copy to avoid changing the original dataset
    home_values_copy = home_values_dataset.copy()
    income_copy = income.copy()
    
    # Manually map state abbreviations to full names
    abbr_to_full = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
        'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
        'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
        'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
        'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
        'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
        'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
        'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
        'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
        'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
        'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Map full state names
    home_values_copy['State'] = home_values_copy['StateName'].map(abbr_to_full)
    
    # Convert to long format and extract year
    home_long = home_values_copy.melt(
        id_vars=['State'],
        value_vars=home_values_copy.columns[5:-1],  # exclude trailing columns if any
        var_name='Date',
        value_name='HomeValue'
    )
    home_long['Year'] = pd.to_datetime(home_long['Date']).dt.year
    home_yearly = home_long.groupby(['State', 'Year'])['HomeValue'].mean().reset_index()
    
    # Clean income data (on copy)
    for col in income_copy.columns[1:]:
        income_copy[col] = income_copy[col].str.replace(',', '').astype(float)
    income_long = income_copy.melt(id_vars=['State'], var_name='Year', value_name='MedianIncome')
    income_long['Year'] = income_long['Year'].astype(int)
    
    # Merge and calculate affordability
    merged = pd.merge(income_long, home_yearly, on=['State', 'Year'], how='inner')
    merged['PriceToIncomeRatio'] = merged['HomeValue'] / merged['MedianIncome']
    return merged

def Regression(clustering_df):
    X = clustering_df[['Income_2023', 'Income_Growth', 'HomeValue_Growth']]
    y = clustering_df['HomeValue_2023']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    baseline_pred = np.full_like(y_test, y_train.mean())
    baseline_r2 = r2_score(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    print(f"Baseline R²: {baseline_r2:.3f}")
    print(f"Baseline RMSE: ${baseline_rmse:,.2f}")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"Linear Regression Model R² Score: {r2:.3f}")
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Linear Regression Model RMSE: ${rmse:,.2f}")
    plt.figure(figsize=(6, 3))
    plt.scatter(y_test, y_pred, color='royalblue', alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Home Values (2023)')
    plt.ylabel('Predicted Home Values (2023)')
    plt.title('Predicted vs Actual Home Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visual1(merged):
    national_trends = merged.groupby('Year').agg({
    'MedianIncome': 'mean',
    'HomeValue': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(8, 4))
    plt.plot(national_trends['Year'], national_trends['MedianIncome'], label='National Median Income', marker='o')
    plt.plot(national_trends['Year'], national_trends['HomeValue'], label='National Median Home Value', marker='s')
    plt.title("U.S. National Median Income vs Home Value (2000–2023)")
    plt.xlabel("Year")
    plt.ylabel("USD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visual3():
    
    hv_data = pd.read_csv("homevalue.csv")
    hv_data.columns.values[5:] = pd.to_datetime(hv_data.columns[5:])
    threshold = 0.3
    home_val = hv_data.loc[hv_data.isnull().mean(axis=1) < threshold]
    ts = home_val.iloc[:, 5:]

   
    long_df = pd.melt(ts, var_name='Date', value_name='HomeValue')
    long_df['Date'] = pd.to_datetime(long_df['Date'])
    long_df['Month'] = long_df['Date'].dt.month
    long_df = long_df.dropna()

  
    avg = long_df.groupby('Month')['HomeValue'].mean()
    mean = avg.mean()
    deviation = (avg - mean) / mean
    labels = deviation.apply(lambda x: 'Best' if x < -0.01 else ('Worst' if x > 0.01 else 'Neutral'))
  
    plot_df = pd.DataFrame({
        'Month': avg.index,
        'AveragePrice': avg.values,
        'Deviation': deviation.values,
        'Label': labels.values
    })

   
    plt.figure(figsize=(10, 5))
    sns.barplot(data=plot_df, x='Month', y='Deviation', hue='Label', dodge=False,
                palette={'Best': 'green', 'Neutral': 'gold', 'Worst': 'red'})
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Which Month is Best to Buy a House?')
    plt.xlabel('Month')
    plt.ylabel('Monthly Price Difference vs Yearly Average (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.xticks(ticks=range(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Buying Time')
    plt.tight_layout()
    plt.grid(True, axis='y')
    plt.show()