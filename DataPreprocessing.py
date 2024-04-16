#MISSING VALUES CORRECTION

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv("your_csv_file")

missing_values = df.isna().sum()

print("Missing values in DataFrame Before Imputing:")
print(missing_values)

# Label encoding for each feature
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])

# Missing value detection and correction for each feature
for col in df.columns:
    imputer = SimpleImputer(strategy='mean')
    df[[col]] = imputer.fit_transform(df[[col]])

missing_values = df.isna().sum()

print("Missing values in DataFrame After Imputing:")
print(missing_values)

#-------------------------------------------------------------------------------
#DIMENSIONALITY REDUCTION

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv("your_csv_file")

# Label encoding for each feature
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])

# Dimensionality reduction for each feature
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(df_pca, columns=['PCA1', 'PCA2'])

print("\nDimensionality Reduced DataFrame:")
print(df_pca)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

plt.bar(range(1, 3), explained_variance_ratio, alpha=0.8, align='center', label='Individual explained variance')
plt.step(range(1, 3), cumulative_variance_ratio, where='mid', label='Cumulative explained variance')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.show()

#-----------------------------------------------------------------------------
#BINNING

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv("your_csv_file")

# Label encoding for each feature
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])

# Binning for each feature
bin_discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
df_binned = bin_discretizer.fit_transform(df)
df_binned = pd.DataFrame(df_binned, columns=[f'{col}_binned' for col in df.columns])

print("\nBinned DataFrame:")
print(df_binned)

#---------------------------------------------------------------------------------------
#MIN-MAX NORMALIZATION

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer


df = pd.read_csv("your_csv_file")

# Label encoding for each feature
for col in df.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    df[col] = label_encoder.fit_transform(df[col])

# Min-max normalization for each feature
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(df_normalized, columns=[f'{col}_normalized' for col in df.columns])

print("\nMin-Max Normalized DataFrame:")
print(df_normalized)
