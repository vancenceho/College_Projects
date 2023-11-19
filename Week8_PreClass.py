# WEEK 8 PROBLEM SET - PRECLASS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_url = "https://ucc0f23537c5713d97f9d527e07e.dl.dropboxusercontent.com/cd/0/inline/CGjGo5qR7n_lrCudx2SL1F2vtuOii8BmWg-DlMcCMluudZ2jG8fLx1YrctvGoCJWuZhUXjWVSSs3iJUAn6RZxa-PQw-if4A4m-ZFrI8kd876M4N0Dj3pKc74GFTEik48BQVvETER5bB884d8do5Fk38V/file#"
df = pd.read_csv(file_url)

# we get a Series data type here
print("Get resale price here only:")
print(df["resale_price"])
print(type(df["resale_price"]))
print("----------------------------")

# using .loc to access a column
print(df.loc[:, "resale_price"])
print(type(df.loc[:, "resale_price"]))
print("-----------------------------")

# using .loc to access the first row
print(df.loc[0, :])
print(type(df.loc[0, :]))
print("-----------------------------")

# cast to DataFrame
print("Get first row of df only as a DataFrame:")
df_row0 = pd.DataFrame(df.loc[0, :])
print(df_row0)
print(type(df_row0))
print("-----------------------------")

# getting rows and columns as DataFrame
df.new = df.loc[0:10, :]
print(df.new)
print(type(df.new))
print("-----------------------------")

# select some columns only
columns = ["town", "block", "resale_price"]
print(df.loc[:, columns])
print("------------------------------")

# similar to above but with 10 rows only
print(df[columns])
print(df.loc[0:10, columns])
print("-------------------------------")

# getting by column index
columns = [1, 3, -1]
df_out = df.iloc[0:10, columns]
print(df_out)
print(type(df_out))
print("---------------------------------")

# selecting data using conditions
columns = ["town", "block", "resale_price"]
df_out = df.loc[:, columns]
df_condition = df_out.loc[df_out["resale_price"] > 500_000, columns]
print(df_condition)
df_condition2 = df_out.loc[(df_out["resale_price"] >= 500_000) & (df_out['resale_price'] <= 600_000), columns]
print(df_condition2)
print("----------------------------------")

# selecting only ANG MO KIO & block between 300s and 400s
df_condition3 = df_out.loc[(df_out["resale_price"] >= 500_000) & (df_out["resale_price"] <= 600_000) & 
                           (df_out["town"] == "ANG MO KIO") & (df_out["block"] >= "300") & (df_out["block"] < "500"), columns]
print(df_condition3)
print("-----------------------------------")

# creating DataFrame
price = df['resale_price']
print(isinstance(price, pd.Series)) # True
price_df = pd.DataFrame(price)
print(isinstance(price_df, pd.DataFrame)) # True
print(price_df)
print("------------------------------------")

# creating series
new_series = pd.Series(list(range(2,101)))
print(isinstance(new_series, pd.Series)) # True
print(new_series)
print("------------------------------------")

# copying
df_1 = df  # shallow copy of df
df_2 = df.copy # deep copy of df
print(df_1 is df) # True
print(df_2 is df) # False

# statistical functions
print(df.describe())
print(df['resale_price'].mean()) # 446724.22886801313
print(df['resale_price'].std()) #  155297.43748684428
print(df['resale_price'].min()) # 140000.0
print(df['resale_price'].max()) # 1258000.0
print(df['resale_price'].quantile(q=0.75)) # 525000.0
print("------------------------------------")

# getting mean over numeric columns only
# print("Getting mean over all columns:") ### will have warning
# mean = df.mean(axis=1)
# print(mean)

print("Getting mean over numeric column only explicitly")
numeric_columns = ["floor_area_sqm", "lease_commence_date", "resale_price"]
mean_numeric = df[numeric_columns].mean(axis=1)
print(mean_numeric)
print("-------------------------------------")

# transposing data frame
df_row0 = pd.DataFrame(df.loc[0, :])
df_row0_transposed = df_row0.T

print("Original:")
print(df_row0)
print("-------------------------------------")
print("After transpose:")
print(df_row0_transposed)
print("-------------------------------------")

# vector operations
def divide_1000(data):
    return data / 1000
df["resale_price_in1000"] = df["resale_price"].apply(divide_1000)
df["resale_price_in1000"]
print("-------------------------------------")
# using lambda to the same function as above
df["resale_price_in1000"] = df["resale_price"].apply(lambda data: data/1000)
print(df["resale_price_in1000"])
df["pricey"] = df["resale_price_in1000"].apply(lambda price: 1 if price > 500 else 0)
print(df[["resale_price_in1000", "pricey"]])
print("--------------------------------------")

# normalization
print(df["floor_area_sqm"].describe())
print(df["lease_commence_date"].describe())
print("---------------------------------------")

# visualization 
# Check the resale price around Tampines. 
# Find out what towns are listed in the dataset
print(np.unique(df["town"]))

# get the dataset for tampines only
df_tampines = df.loc[df["town"] == "TAMPINES", :]
print(df_tampines)

# plot distribution using histplot
sns.histplot(x="resale_price", data=df_tampines)
plt.show()
# to show it vertically
sns.set()
myplot = sns.histplot(y="resale_price", data=df_tampines, bins=10)
myplot.set_xlabel("Count", fontsize=16)
myplot.set_ylabel("Resale Price", fontsize=16)
myplot.set_title("HDB Resale Price in Tampines", fontsize=16)
plt.show()

myplot = sns.histplot(y="resale_price_in1000", hue="storey_range", multiple="stack", data=df_tampines, bins=10)
myplot.set_xlabel("Count", fontsize=16)
myplot.set_ylabel("Resale Price in $1000", fontsize=16)
myplot.set_title("HDB Resale Price in Tampines", fontsize=16)
plt.show()

# plot descriptive using boxplot
sns.boxplot(x="resale_price", data=df_tampines)
plt.show()

# plot relationships using scatterplot
myplot = sns.scatterplot(x="floor_area_sqm", y="resale_price_in1000", hue="flat_type", data=df_tampines)
myplot.set_xlabel("Floor Area ($m^2$)")
myplot.set_ylabel("Resale Price in $1000")
plt.show()

# plot realtionships on multiple columns 
myplot = sns.pairplot(data=df_tampines)
plt.show()
