import pandas as pd
from IPython.display import display, HTML

country_iso_df = pd.read_csv("F:\Workspace\country_codes.csv")
country_iso_df = country_iso_df[["name", "alpha-3"]]

# Add additional alternatives
additional = [["Ivory Coast", "CIV"], ["Cote dIvoire", "CIV"], ['Bolivia', 'BOL'], ['Congo (Brazzaville)', 'COG'], ['Congo (Kinshasa)', 'COD'], 
              ['Congo (Democratic Republic of the)', 'COD'], ['Congo (Dem. Rep.)', 'COD'], ["Czech Republic", "CZE"], ['Hong Kong S.A.R. of China', 'HKG'], 
              ['Hong Kong, China (SAR)', 'HKG'], ['Iran', 'IRN'], ['Laos', 'LAO'], ['Moldova', 'MDA'], ['Moldova (Republic of)', 'MDA'], ['Russia', 'RUS'], 
              ['South Korea', 'KOR'], ['Korea (Republic of)', 'KOR'], ["Korea (Democratic People's Rep. of)", 'PRK'], ['State of Palestine', 'PSE'], ['Syria', 'SYR'], 
              ['Taiwan Province of China', 'TWN'], ['Tanzania', 'TZA'], ['Tanzania (United Republic of)', 'TZA'], ['Turkiye', 'TUR'], ['Türkiye', 'TUR'], 
              ['United Kingdom', 'GBR'], ['United States', 'USA'], ['Venezuela', 'VEN'], ['Vietnam', 'VNM'], ['Eswatini (Kingdom of)', 'SWZ'], 
              ['Afghanistan, Islamic Republic of', 'AFG']
]

for item in additional: country_iso_df.loc[len(country_iso_df.index)] = item 
#display(country_iso_df.head())

def insert_iso(df):
    # Merging to match "Country" to "ISO3"
    df = pd.merge(df, country_iso_df, how='left', left_on='Country', right_on='name')
    df.loc[df['name'] == df['Country'], 'ISO3'] = df['alpha-3']
    df.drop(["name", "alpha-3"], axis=1, inplace=True)
    # Reordering
    order = ["Country", "ISO3"] + sorted(df.columns)[:-2]
    df = df[order]
    return df

def insert_buffer_columns(df, years=range(2013, 2023)):
    for year in years:
        year = str(year)
        if year not in df.columns: df[year] = None
            
def interpolate_country(group, col: str): 
    # Any values bounded by a previous and next value Eg. b/w 2010 -> 2015
    group[col].interpolate(method="linear", inplace=True)
    # Any remaining values (only has a prev value) Eg. use 2021 -> 2022
    group[col].interpolate(method="ffill", inplace=True)
    # Any remaining values (only has a next value) Eg. use 2015 -> 2013
    group[col].interpolate(method="bfill", inplace=True)
    group[col] = round(group[col], 3)
    return group
            
def interpolate_missing(df: pd.DataFrame, is_long_format: bool, col: str) -> pd.DataFrame:
    # df must be in long format!
    
    if not is_long_format:
        # Converting into long format
        df = pd.melt(df, id_vars=['Country', 'ISO3'], var_name='Year', value_name=col)
        
    # Explicit type + downcasting into more memory-efficient datatypes where possible
    # and enable float for interpolation 
    df["Country"] = df["Country"].astype("category")
    df["ISO3"] = df["ISO3"].astype("category")
    df["Year"] = df["Year"].astype("int16")
    df[col] = df[col].astype("float")

    # Sorting Country, then Year to achieve standardized format and prepare for interpolation
    df.sort_values(by=["Country", "Year"], inplace=True)
    df.reset_index(inplace=True, drop=True)
    
    df = df.groupby('Country').apply(lambda x: interpolate_country(x, col))
    
    # Drop years out of range
    df = df[df['Year'].isin(list(range(2013, 2023)))]
    return df

print(country_iso_df)

# GROSS DOMESTIC PRODUCT (GDP)
gdp_df = pd.read_csv("F:\Workspace\Gross Domestic Product.csv")

# Columns needed.
cols = ["Country", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]

# Dropping columns not needed.
gdp_df = gdp_df[cols]

# Melting the DataFrame to convert from wide to long format.
gdp_df_sorted = pd.melt(gdp_df, id_vars=['Country'], var_name="Year", value_name="Gross Domestic Product (GDP)")

# Merge with ISO DataFrame to add ISO with respect to the countries. 
gdp_df_merged = gdp_df_sorted.merge(country_iso_df, left_on="Country", right_on="name", how="left")

# Drop the redudant "name" column if needed. 
gdp_df_merged = gdp_df_merged.drop(columns=["name"])

# Sorting the Dataframe by "Country" column.
gdp_df_final = gdp_df_merged.sort_values(by=["Country", "Year"], ascending=[True, True])

# Rename the alpha-3
gdp_df_final.rename(columns={"alpha-3": "ISO3"}, inplace=True)

# Reorder the dataframe to the correct order.
gdp_df_final = gdp_df_final.reindex(columns=['Country', 'ISO3', 'Year', 'Gross Domestic Product (GDP)'])

# Interpolate missing values. 
print(gdp_df_final)
# Extract required data
#gdp_df_final.to_csv("GDP.csv", sep=',', index=False, encoding='utf-8')


# POPULATION 
pop_df = pd.read_csv("F:\Workspace\Population.csv")
print(pop_df.head())

pop_col = ["Country Name", "Country Code", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022"]
pop_df = pop_df[pop_col]

# Renaming necessary columns.
pop_df.rename(columns={'Country Name': 'Country', 'Country Code': 'ISO3'}, inplace=True)

# Melting the DataFrame to convert to long format.
pop_col_final = pd.melt(pop_df, id_vars=['Country', 'ISO3'], var_name='Year', value_name='Population')

print(pop_df.head())
# Extract the required data.
#pop_col_final.to_csv("Population_Cleaned.csv", sep=',', index=False, encoding='utf-8')


# EDUCATION POPULATION INDEX (EDI)
epi_2012 = pd.read_excel("F:\Workspace/2012-epi.xls")
epi_2014 = pd.read_excel("F:\Workspace/2014-epi.xls")
epi_2016 = pd.read_excel("F:\Workspace/2016-epi.xlsx")
epi_2018 = pd.read_excel("F:\Workspace/2018-epi.xlsx")
epi_2020 = pd.read_excel("F:\Workspace/2020-epi.xlsx")
epi_2022 = pd.read_excel("F:\Workspace/2022-epi.xlsx")

# 2012
epi_2012_col = ['Country', 'ISO3V10', 'EPI']
epi_2012 = epi_2012[epi_2012_col]
epi_2012['Year'] = 2012
epi_2012_df = epi_2012.copy()

epi_2012_df.rename(columns={'ISO3V10': 'ISO3', 'EPI': 'Education Population Index'}, inplace=True)

# 2014
epi_2014_col = ['Country', 'ISO3v10', '2014 EPI Score']
epi_2014 = epi_2014[epi_2014_col]
epi_2014["Year"] = 2014
epi_2014_df = epi_2014.copy()

epi_2014_df.rename(columns={'ISO3v10': 'ISO3', '2014 EPI Score': 'Education Population Index'}, inplace=True)

# 2016
epi_2016_col = ['Country', 'ISO3', '2016 EPI Score']
epi_2016 = epi_2016[epi_2016_col]
epi_2016['Year'] = 2016
epi_2016_df = epi_2016.copy()

epi_2016_df.rename(columns={'ISO3': 'ISO3', '2016 EPI Score': 'Education Population Index'}, inplace=True)

# 2018
epi_2018_col = ['iso', 'country', 'EPI.current']
epi_2018 = epi_2018[epi_2018_col]
epi_2018['Year'] = 2018
epi_2018_df = epi_2018.copy()

epi_2018_df.rename(columns={'iso': 'ISO3', 'country': 'Country', 'EPI.current': 'Education Population Index'}, inplace=True)

# 2020
epi_2020_col = ['iso', 'country', 'EPI.new']
epi_2020 = epi_2020[epi_2020_col]
epi_2020['Year'] = 2020
epi_2020_df = epi_2020.copy()

epi_2020_df.rename(columns={'iso': 'ISO3', 'country': 'Country', 'EPI.new': 'Education Population Index'}, inplace=True)

# 2022
epi_2022_col = ['iso', 'country', 'EPI.new']
epi_2022 = epi_2022[epi_2022_col]
epi_2022['Year'] = 2022
epi_2022_df = epi_2022.copy()

epi_2022_df.rename(columns={'iso': 'ISO3', 'country': 'Country', 'EPI.new': 'Education Population Index'}, inplace=True)

# Final
epi_df = pd.concat([epi_2012_df, epi_2014_df, epi_2016_df, epi_2018_df, epi_2020_df, epi_2022_df], ignore_index=True)
epi_df = epi_df.sort_values(by=['Country', 'Year'], ascending=[True, True])
cols = ['Country', 'ISO3', 'Year', 'Education Population Index']
epi_df = epi_df[cols]

print(epi_df)
#epi_df.to_csv("Education Population Index.csv", sep=',', index=False, encoding='utf-8')


# ND-GAIN COUNTRY INDEX
ngci_df = pd.read_csv(r"F:\Workspace/ND-Gain Country Index.csv")
ngci_col = ['ISO3', 'Name', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021']
ngci_df = ngci_df[ngci_col]

ngci_df_original = ngci_df.copy()

ngci_df_original.rename(columns={'ISO3': 'ISO3', 'Name': 'Country'}, inplace=True)

ngci_final = pd.melt(ngci_df_original, id_vars=['Country', 'ISO3'], var_name='Year', value_name='ND-Gain Country Index')
ngci_final = ngci_final.sort_values(by=['Country', 'Year'], ascending=[True, True])

print(ngci_final)
#ngci_final.to_csv('ND Gain Country Index.csv', sep=',', index=False, encoding='utf-8')


# CONSUMER PRICE INDEX
cpi_df = pd.read_excel("F:\Workspace\Consumer Price Index.xlsx")

# Category = Food and non-alcoholic beverages
cpi_columns = ['Country', 'Category', 
               '2013M01', '2013M02', '2013M03', '2013M04', '2013M05', '2013M06', '2013M07', '2013M08', '2013M09', '2013M10', '2013M11', '2013M12',
               '2014M01', '2014M02', '2014M03', '2014M04', '2014M05', '2014M06', '2014M07', '2014M08', '2014M09', '2014M10', '2014M11', '2014M12',
               '2015M01', '2015M02', '2015M03', '2015M04', '2015M05', '2015M06', '2015M07', '2015M08', '2015M09', '2015M10', '2015M11', '2015M12',
               '2016M01', '2016M02', '2016M03', '2016M04', '2016M05', '2016M06', '2016M07', '2016M08', '2016M09', '2016M10', '2016M11', '2016M12',
               '2017M01', '2017M02', '2017M03', '2017M04', '2017M05', '2017M06', '2017M07', '2017M08', '2017M09', '2017M10', '2017M11', '2017M12',
               '2018M01', '2018M02', '2018M03', '2018M04', '2018M05', '2018M06', '2018M07', '2018M08', '2018M09', '2018M10', '2018M11', '2018M12',
               '2019M01', '2019M02', '2019M03', '2019M04', '2019M05', '2019M06', '2019M07', '2019M08', '2019M09', '2019M10', '2019M11', '2019M12',
               '2020M01', '2020M02', '2020M03', '2020M04', '2020M05', '2020M06', '2020M07', '2020M08', '2020M09', '2020M10', '2020M11', '2020M12',
               '2021M01', '2021M02', '2021M03', '2021M04', '2021M05', '2021M06', '2021M07', '2021M08', '2021M09', '2021M10', '2021M11', '2021M12',
               '2022M01', '2022M02', '2022M03', '2022M04', '2022M05', '2022M06', '2022M07', '2022M08', '2022M09', '2022M10', '2022M11', '2022M12']
cpi_food_df1 = cpi_df.drop(cpi_df[cpi_df['Category'] != 'Food and non-alcoholic beverages'].index)
cpi_food_df1 = cpi_food_df1[cpi_columns]
cpi_food_df1 = cpi_food_df1.fillna(0)

c_2013 = ['2013M01', '2013M02', '2013M03', '2013M04', '2013M05', '2013M06', '2013M07', '2013M08', '2013M09', '2013M10', '2013M11', '2013M12']
c_2014 = ['2014M01', '2014M02', '2014M03', '2014M04', '2014M05', '2014M06', '2014M07', '2014M08', '2014M09', '2014M10', '2014M11', '2014M12']
c_2015 = ['2015M01', '2015M02', '2015M03', '2015M04', '2015M05', '2015M06', '2015M07', '2015M08', '2015M09', '2015M10', '2015M11', '2015M12']
c_2016 = ['2016M01', '2016M02', '2016M03', '2016M04', '2016M05', '2016M06', '2016M07', '2016M08', '2016M09', '2016M10', '2016M11', '2016M12']
c_2017 = ['2017M01', '2017M02', '2017M03', '2017M04', '2017M05', '2017M06', '2017M07', '2017M08', '2017M09', '2017M10', '2017M11', '2017M12']
c_2018 = ['2018M01', '2018M02', '2018M03', '2018M04', '2018M05', '2018M06', '2018M07', '2018M08', '2018M09', '2018M10', '2018M11', '2018M12']
c_2019 = ['2019M01', '2019M02', '2019M03', '2019M04', '2019M05', '2019M06', '2019M07', '2019M08', '2019M09', '2019M10', '2019M11', '2019M12']
c_2020 = ['2020M01', '2020M02', '2020M03', '2020M04', '2020M05', '2020M06', '2020M07', '2020M08', '2020M09', '2020M10', '2020M11', '2020M12']
c_2021 = ['2021M01', '2021M02', '2021M03', '2021M04', '2021M05', '2021M06', '2021M07', '2021M08', '2021M09', '2021M10', '2021M11', '2021M12']
c_2022 = ['2022M01', '2022M02', '2022M03', '2022M04', '2022M05', '2022M06', '2022M07', '2022M08', '2022M09', '2022M10', '2022M11', '2022M12']

cpi_food_df1['2013'] = cpi_food_df1[c_2013].sum(axis=1)
cpi_food_df1['2014'] = cpi_food_df1[c_2014].sum(axis=1)
cpi_food_df1['2015'] = cpi_food_df1[c_2015].sum(axis=1)
cpi_food_df1['2016'] = cpi_food_df1[c_2016].sum(axis=1)
cpi_food_df1['2017'] = cpi_food_df1[c_2017].sum(axis=1)
cpi_food_df1['2018'] = cpi_food_df1[c_2018].sum(axis=1)
cpi_food_df1['2019'] = cpi_food_df1[c_2019].sum(axis=1)
cpi_food_df1['2020'] = cpi_food_df1[c_2020].sum(axis=1)
cpi_food_df1['2021'] = cpi_food_df1[c_2021].sum(axis=1)
cpi_food_df1['2022'] = cpi_food_df1[c_2022].sum(axis=1)

cpi_food_columns = ['Country', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
cpi_food_df = cpi_food_df1[cpi_food_columns]
cpi_food_sorted = pd.melt(cpi_food_df, id_vars=['Country'], var_name='Year', value_name='Consumer Price Index (Food)')
cpi_food_final = cpi_food_sorted.merge(country_iso_df, left_on="Country", right_on="name", how="left")
cpi_food_final = cpi_food_final.drop(columns=["name"])
cpi_food_final.rename(columns={"alpha-3": "ISO3"}, inplace=True)
cpi_food_final = cpi_food_final.reindex(columns=['Country', 'ISO3', 'Year', 'Consumer Price Index (Food)'])
cpi_food_final = cpi_food_final.sort_values(by=['Country', 'Year'], ascending=[True, True])

display(cpi_food_final)
#cpi_food_final.to_csv('Consumer Price Index (Food).csv', sep=',', index=False, encoding='utf-8')


cpi_health_df1 = cpi_df.drop(cpi_df[cpi_df['Category'] != 'Health'].index)
cpi_food_df1 = cpi_food_df1.fillna(0)

cpi_health_df1['2013'] = cpi_health_df1[c_2013].sum(axis=1)
cpi_health_df1['2014'] = cpi_health_df1[c_2014].sum(axis=1)
cpi_health_df1['2015'] = cpi_health_df1[c_2015].sum(axis=1)
cpi_health_df1['2016'] = cpi_health_df1[c_2016].sum(axis=1)
cpi_health_df1['2017'] = cpi_health_df1[c_2017].sum(axis=1)
cpi_health_df1['2018'] = cpi_health_df1[c_2018].sum(axis=1)
cpi_health_df1['2019'] = cpi_health_df1[c_2019].sum(axis=1)
cpi_health_df1['2020'] = cpi_health_df1[c_2020].sum(axis=1)
cpi_health_df1['2021'] = cpi_health_df1[c_2021].sum(axis=1)
cpi_health_df1['2022'] = cpi_health_df1[c_2022].sum(axis=1)

cpi_health_columns = ['Country', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']
cpi_health_df = cpi_health_df1[cpi_health_columns]
cpi_health_sorted = pd.melt(cpi_health_df, id_vars=['Country'], var_name='Year', value_name='Consumer Price Index (Health)')
cpi_health_final = cpi_health_sorted.merge(country_iso_df, left_on="Country", right_on="name", how="left")
cpi_health_final = cpi_health_final.drop(columns=["name"])
cpi_health_final.rename(columns={"alpha-3": "ISO3"}, inplace=True)
cpi_health_final = cpi_health_final.reindex(columns=['Country', 'ISO3', 'Year', 'Consumer Price Index (Health)'])
cpi_health_final = cpi_health_final.sort_values(by=['Country', 'Year'], ascending=[True, True])

display(cpi_health_final)
#cpi_health_final.to_csv('Consumer Price Index (Health).csv', sep=',', index=False, encoding='utf-8')