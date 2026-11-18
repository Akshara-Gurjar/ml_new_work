import pandas as pd
import numpy as np

df=pd.read_csv("C:\\Users\\bhagw\\OneDrive\\Desktop\\ml_new_work\\insurance.csv")


print(df.head(3))

# label encoding
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
df["sex"]=le.fit_transform(df["sex"])
df["smoker"]=le.fit_transform(df["smoker"])
df["region"]=le.fit_transform(df["region"])
print(df.head(2))


# one hot encoding
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
df=pd.read_csv("C:\\Users\\bhagw\\OneDrive\\Desktop\\ml_new_work\\insurance.csv")


ohe=OneHotEncoder(sparse_output=False,handle_unknown="ignore")

encoded=ohe.fit_transform(df[["sex","smoker","region"]])
encoded_df=pd.DataFrame(encoded,
                        columns=ohe.get_feature_names_out(["sex","smoker","region"]))


#remove original categorical columns
df=df.drop(["sex","smoker","region"],axis=1)


df=pd.concat([df,encoded_df],axis=1)

print("after one hot encoding")
print(df.head(2))
