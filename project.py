import pandas as pd
import numpy as np
import scipy.stats
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# get rid of warnings
import warnings
warnings.filterwarnings("ignore")
# get more than one output per Jupyter cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# for functions we implement later
from utils import best_fit_distribution
from utils import plot_result



df = pd.read_csv("health_data.csv")


df.shape
df.head()


df.drop(columns=["PatientID", "Name"], inplace=True) # dropped because unique for every row
df.drop(columns=["RoomNum", "docRef"], inplace=True) # dropped because almost unique for every row
df.dropna(inplace=True)



df.shape
df.head()



encoders = [(["Sex"], LabelEncoder()), (["Condition"], LabelEncoder())]
mapper = DataFrameMapper(encoders, df_out=True)
new_cols = mapper.fit_transform(df.copy())
df = pd.concat([df.drop(columns=["Sex", "Condition"]), new_cols], axis="columns")



df.shape
df.head()


df.nunique()


categorical = []
continuous = []

for c in list(df):
    col = df[c]
    nunique = col.nunique()
    if nunique < 20:
        categorical.append(c)
    else:
        continuous.append(c)




for c in categorical:
    counts = df[c].value_counts()
    np.random.choice(list(counts.index), p=(counts/len(df)).values, size=5)


best_distributions = []




for c in continuous:
    data = df[c]
    best_fit_name, best_fit_params = best_fit_distribution(data, 50)
    best_distributions.append((best_fit_name, best_fit_params))



# Result
best_distributions = [
    ('fisk', (11.744665309421649, -66.15529969956657, 94.73575225186589)),
    ('halfcauchy', (-5.537941926133496e-09, 17.86796415175786))]


plot_result(df, continuous, best_distributions)


def generate_like_df(df, categorical_cols, continuous_cols, best_distributions, n, seed=0):
    np.random.seed(seed)
    d = {}

    for c in categorical_cols:
        counts = df[c].value_counts()
        d[c] = np.random.choice(list(counts.index), p=(counts/len(df)).values, size=n)

    for c, bd in zip(continuous_cols, best_distributions):
        dist = getattr(scipy.stats, bd[0])
        d[c] = dist.rvs(size=n, *bd[1])

    return pd.DataFrame(d, columns=categorical_cols+continuous_cols)




gendf = generate_like_df(df, categorical, continuous, best_distributions, n=100)
gendf.shape
gendf.head()



gendf.columns = list(range(gendf.shape[1]))



gendf.to_csv("output.csv", index_label="id")


