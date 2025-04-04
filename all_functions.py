import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
    
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
    state_to_abbrev = {
        'Alabama': 'AL','Alaska': 'AK','Arizona': 'AZ','Arkansas': 'AR','California': 'CA',
        'Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','Florida': 'FL','Georgia': 'GA','Hawaii': 'HI',
        'Idaho': 'ID', 'Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS', 'Kentucky': 'KY',
        'Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA',
        'Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO',
        'Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH',
        'New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Ohio': 'OH',
        'Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA',
        'Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN',
        'Texas': 'TX','Utah': 'UT','Vermont': 'VT','Virginia': 'VA',
        'Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'
    }
    
    
    df = median_income.copy()
    
    df['State_Abbrev'] = df['State'].map(state_to_abbrev)
    df_clean = df.dropna(subset=['State_Abbrev'])
    dates = df_clean.columns[1:]
    date_columns = df_clean.columns.difference(['State', 'State_Abbrev'])
    
    df_long = df_clean.melt(
        id_vars=['State', 'State_Abbrev'],
        value_vars=date_columns,
        var_name='Date',
        value_name='Value'
    )
    
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
    
    
    fig.update_layout(width=1000, height=700)
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

    heatmap_data = merged.pivot(index='State', columns='Year', values='PriceToIncomeRatio')
    plt.figure(figsize=(14, 12))
    sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.5, linecolor='gray', cbar_kws={'label': 'Price-to-Income Ratio'})
    plt.title("Heatmap of Housing Affordability by State and Year")
    plt.xlabel("Year")
    plt.ylabel("State")
    plt.tight_layout()
    plt.show()




    




