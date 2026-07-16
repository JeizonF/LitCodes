import pandas as pd
import numpy as np


def features(df):

    c1=df["canal1"].values
    c2=df["canal2"].values

    return [

        np.mean(c1),
        np.std(c1),
        np.max(c1),
        np.min(c1),

        np.mean(c2),
        np.std(c2),
        np.max(c2),
        np.min(c2),

        np.ptp(c1),
        np.ptp(c2),

        np.sqrt(np.mean(c1**2)),
        np.sqrt(np.mean(c2**2)),

        np.mean(np.abs(np.diff(c1))),
        np.mean(np.abs(np.diff(c2)))

    ]



# normal

normal=pd.read_csv(
"dados/openBCI_raw_2018-07-02_15-46-30.txt",
skiprows=5
)


normal=normal.iloc[:,:3]

normal.columns=[
"amostra",
"canal1",
"canal2"
]


normal=normal.apply(
pd.to_numeric,
errors="coerce"
)

normal=normal.dropna()



# piscada

p=pd.read_csv(
"dados/piscada.csv",
header=None
)


p=p.iloc[4:,:3]

p.columns=[
"tempo",
"canal1",
"canal2"
]


p["canal1"]=pd.to_numeric(
p["canal1"],
errors="coerce"
)


p["canal2"]=pd.to_numeric(
p["canal2"],
errors="coerce"
)


p=p.dropna()


p["canal1"]=p["canal1"]*1000
p["canal2"]=p["canal2"]*1000



print("\nNORMAL")
print(features(normal.iloc[:100]))


print("\nPISCADA")
print(features(p.iloc[:100]))