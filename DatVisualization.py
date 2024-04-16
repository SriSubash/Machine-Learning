import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Sample Data

data = pd.read_csv("your_csv_file")
display(data.head(10))

#Scatter Plot

data = pd.read_csv("C:/Users/subas/Downloads/Lumpy skin disease data.csv",nrows=5000)
plt.scatter(data['region'].astype(str), data['dominant_land_cover'].astype(str))
plt.title("Scatter Plot")
plt.xlabel('Region')
plt.ylabel('Dominant_Land_Cover')
plt.show()

#Line Chart

data = pd.read_csv("C:/Users/subas/Downloads/Lumpy skin disease data.csv",nrows=2000)
plt.plot(data['cld'])
plt.plot(data['pre'])
plt.title("Scatter Plot")
plt.xlabel('Cloud')
plt.ylabel('Precipitation')
plt.show()

#Bar Chart

data = pd.read_csv("C:/Users/subas/Downloads/Lumpy skin disease data.csv")
plt.bar(data['region'].astype(str), data['dominant_land_cover'].astype(str))
plt.title("Bar Chart")
plt.xlabel('Region')
plt.ylabel('Dominant_Land_Cover')
plt.show()

#Histogram

plt.hist(data['dominant_land_cover'].astype(str))
plt.title("Histogram")
plt.show()

#Histo Plot

sns.histplot(x='tmp', data=data, kde=True, hue='lumpy')
plt.show()

#Line Plot

sns.lineplot(x="lumpy", y="dominant_land_cover", data=data)
plt.title('Line Plot')
plt.show()

#Box Plot

sns.boxplot(x="region", y="tmp", data=data)

#FacetGrid

sns.FacetGrid(data, hue="region") \
   .map(sns.kdeplot, "pre") \
   .add_legend()

#Heatmap

sns.heatmap(data.corr(), annot=True)

#Scatter Matrix

from pandas.plotting import scatter_matrix
fig, ax = plt.subplots(figsize=(12,12))
scatter_matrix(data, alpha=1, ax=ax)
data = pd.read_csv("C:/Users/subas/Downloads/Lumpy skin disease data.csv",nrows=1000)
sns.pairplot(data)
