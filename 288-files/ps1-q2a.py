#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import requests
import os

# Read the data
df = pd.read_csv('../data/dhs_clusters.csv')

# Convert year to datetime for better plotting
df['date'] = pd.to_datetime(df['year'], format='%Y')

# Group by year and calculate the average IWI
iwi_by_year = df.groupby('year')['iwi'].mean().reset_index()

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(iwi_by_year['year'], iwi_by_year['iwi'], marker='o', linestyle='-', color='blue')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Average Monthly IWI')
plt.title('Average Monthly IWI Over Time')
plt.grid(True, linestyle='--', alpha=0.7)

# Format the x-axis to show years properly with rotation to avoid overlap
plt.xticks(iwi_by_year['year'], rotation=45)

# Add data points
for x, y in zip(iwi_by_year['year'], iwi_by_year['iwi']):
    plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("yearly-iwi.png")
plt.show()

# %%
# Get unique countries in the dataset
unique_countries = df['country'].unique()

# Print the list of countries
print("Countries in the dataset:")
for country in sorted(unique_countries):
    print(f"- {country}")

# Count the number of countries
num_countries = len(unique_countries)
print(f"\nTotal number of countries: {num_countries}")

# Optional: Create a bar chart showing the number of clusters per country
country_counts = df['country'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
bars = plt.bar(country_counts.index, country_counts.values, color='skyblue')
plt.xlabel('Country')
plt.ylabel('Number of Clusters')
plt.title('Number of Clusters by Country')
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add count labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("clusters-by-country.png")
plt.show()

# %%
# Create side-by-side plots of the average IWI per year for Egypt, Nigeria, and Malawi
countries_to_plot = ['egypt', 'nigeria', 'malawi']

plt.figure(figsize=(15, 6))

for i, country in enumerate(countries_to_plot, 1):
    # Filter data for the current country
    country_data = df[df['country'] == country]
    
    # Group by year and calculate mean IWI
    country_iwi_by_year = country_data.groupby('year')['iwi'].mean().reset_index()
    
    # Create subplot
    plt.subplot(1, 3, i)
    
    # Plot the data
    plt.plot(country_iwi_by_year['year'], country_iwi_by_year['iwi'], 
             marker='o', linestyle='-', linewidth=2, markersize=8)
    
    # Add labels and title
    plt.xlabel('Year')
    plt.ylabel('Average IWI')
    plt.title(f'Average IWI Over Time - {country.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Format x-axis
    plt.xticks(country_iwi_by_year['year'], rotation=45)
    
    # Add data points
    for x, y in zip(country_iwi_by_year['year'], country_iwi_by_year['iwi']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig("iwi-egypt-nigeria-malawi.png")
plt.show()

# %%
# Create a map that plots the IWI values using latitude and longitude, and label each country
plt.figure(figsize=(15, 10))

# Filter data for years before 2010
df_before_2010 = df[df['year'] < 2010]

# Create a scatter plot with IWI values for years before 2010
scatter = plt.scatter(df_before_2010['lon'], df_before_2010['lat'], 
                     c=df_before_2010['iwi'], 
                     cmap='viridis', 
                     alpha=0.5,
                     s=5,  # Size of points
                     edgecolor='k',  # Black edge for better visibility
                     linewidth=0.5)

# Add a colorbar to show the IWI scale
cbar = plt.colorbar(scatter)
cbar.set_label('International Wealth Index (IWI)')

# Calculate the mean position for each country to place labels
country_positions = df_before_2010.groupby('country').agg({
    'lat': 'mean',
    'lon': 'mean'
}).reset_index()

# Add country labels at the mean position of each country's data points
for _, row in country_positions.iterrows():
    plt.text(row['lon'], row['lat'], row['country'].capitalize(), 
             fontsize=9, 
             ha='center', 
             va='center', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of International Wealth Index (IWI) Before 2010')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.4)

# Adjust layout and save
plt.tight_layout()
plt.savefig("iwi-geographic-distribution-before-2010.png", dpi=300)
plt.show()

# %%
# Create a map that plots the IWI values using latitude and longitude for years after 2010
plt.figure(figsize=(15, 10))

# Filter data for years after 2010
df_after_2010 = df[df['year'] >= 2010]

# Create a scatter plot with IWI values for years after 2010
scatter = plt.scatter(df_after_2010['lon'], df_after_2010['lat'], 
                     c=df_after_2010['iwi'], 
                     cmap='viridis', 
                     alpha=0.5,
                     s=5,  # Size of points
                     edgecolor='k',  # Black edge for better visibility
                     linewidth=0.5)

# Add a colorbar to show the IWI scale
cbar = plt.colorbar(scatter)
cbar.set_label('International Wealth Index (IWI)')

# Calculate the mean position for each country to place labels
country_positions = df_after_2010.groupby('country').agg({
    'lat': 'mean',
    'lon': 'mean'
}).reset_index()

# Add country labels at the mean position of each country's data points
for _, row in country_positions.iterrows():
    plt.text(row['lon'], row['lat'], row['country'].capitalize(), 
             fontsize=9, 
             ha='center', 
             va='center', 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Geographic Distribution of International Wealth Index (IWI) After 2010')

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.4)

# Adjust layout and save
plt.tight_layout()
plt.savefig("iwi-geographic-distribution-after-2010.png", dpi=300)
plt.show()

# %%
# Create a plot showing the number of entries per year
plt.figure(figsize=(12, 6))

# Count the number of entries per year
year_counts = df['year'].value_counts().sort_index()

# Create a bar plot
bars = plt.bar(year_counts.index, year_counts.values, color='steelblue')

# Add labels and title
plt.xlabel('Year')
plt.ylabel('Number of Entries')
plt.title('Number of DHS Cluster Entries per Year')

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(year_counts.values),
             f'{height}',
             ha='center', va='bottom', fontsize=9)

# Adjust x-axis to show all years
plt.xticks(year_counts.index, rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig("entries-per-year.png", dpi=300)
plt.show()

# %%
# Create a scatter plot showing IWI vs number of households
plt.figure(figsize=(10, 6))

# Check if there's data for Nigeria in 2014
nigeria_2014 = df[(df['country'] == 'nigeria') & (df['year'] == 2014)]

# If no data for Nigeria in 2014, use all countries or a different year
if len(nigeria_2014) == 0:
    # Try all countries in 2014 or use a different filter based on available data
    plot_data = df[df['year'] >= 2010]  # Using data after 2010
    title = 'IWI vs Number of Households (After 2010)'
else:
    plot_data = nigeria_2014
    title = 'IWI vs Number of Households in Nigeria (2014)'

# Create the scatter plot
plt.scatter(plot_data['households'], plot_data['iwi'], 
            alpha=0.7, c='steelblue', s=50)

# Add labels and title
plt.xlabel('Number of Households')
plt.ylabel('International Wealth Index (IWI)')
plt.title(title)
plt.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig("iwi-vs-households.png", dpi=300)
plt.show()

# %%
# Filter data for Nigeria
nigeria_data = df[df['country'] == 'nigeria']

# Count entries per year for Nigeria
nigeria_year_counts = nigeria_data['year'].value_counts().sort_index()

# Create a bar chart for Nigeria entries per year
plt.figure(figsize=(10, 6))
bars = plt.bar(nigeria_year_counts.index, nigeria_year_counts.values, color='forestgreen')

# Add title and labels
plt.title('Number of Entries per Year for Nigeria', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Entries', fontsize=12)

# Add grid for better readability
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1*max(nigeria_year_counts.values),
             f'{height}',
             ha='center', va='bottom', fontsize=9)

# Adjust x-axis to show all years
plt.xticks(nigeria_year_counts.index, rotation=45)

# Adjust layout and save
plt.tight_layout()
plt.savefig("nigeria-entries-per-year.png", dpi=300)
plt.show()

# %%
# Filter data for Nigeria in 2018
nigeria_2018_data = df[(df['country'] == 'nigeria') & (df['year'] == 2018)]

# Create a scatter plot of IWI vs households for Nigeria in 2018
plt.figure(figsize=(10, 6))
plt.scatter(nigeria_2018_data['households'], nigeria_2018_data['iwi'], 
            alpha=0.7, c='forestgreen', s=50)

# Add labels and title
plt.xlabel('Number of Households', fontsize=12)
plt.ylabel('International Wealth Index (IWI)', fontsize=12)
plt.title('IWI vs Number of Households for Nigeria (2018)', fontsize=14)
plt.grid(True, alpha=0.3)

# Add a best fit line
if len(nigeria_2018_data) > 1:  # Only add trendline if we have enough data points
    z = np.polyfit(nigeria_2018_data['households'], nigeria_2018_data['iwi'], 1)
    p = np.poly1d(z)
    plt.plot(nigeria_2018_data['households'], p(nigeria_2018_data['households']), 
             "r--", alpha=0.8, label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}")
    plt.legend()

# Adjust layout and save
plt.tight_layout()
plt.savefig("nigeria-2018-iwi-vs-households.png", dpi=300)
plt.show()

# %%
