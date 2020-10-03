# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 10:55:35 2020

@author: Chuck
"""

'''rename column name, change case/type'''

import pandas as pd; import numpy as np

df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df=df.rename(str.lower, axis='columns')
df.rename(index=str, inplace=True)
df.index.dtype
df.rename(index=str).index
df

s = pd.Series([1, 2, 3], dtype=np.int64, name='Numbers')
s.dtype
s.rename(index=str).index.dtype


'''LabelEncoder & OneHotEncoder'''
import pandas as pd; import numpy as np
data = {'age':[20,30,40, 50],'rbc':['normal','normal','abnormal','unknown'],'pc':['notpresent','notpresent','present','notpresent']}
df = pd.DataFrame(data, columns=('age', 'rbc', 'pc'))   #if age renamed, dtype will change to 'o'
df.columns
df.age.dtype
df2 = df.copy()
# Categorical boolean mask
categorical_feature_mask = df.dtypes==object
# filter categorical columns using mask and turn it into a list
categorical_cols = df.columns[categorical_feature_mask].tolist()  #['rbc', 'pc']


# import LabelEncoder
from sklearn.preprocessing import LabelEncoder
# instantiate labelencoder object
le = LabelEncoder()
# apply le on categorical feature columns
df[categorical_cols] = df[categorical_cols].apply(lambda col: le.fit_transform(col))
df[categorical_cols].head()


# import OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# instantiate OneHotEncoder
ohe = OneHotEncoder() 
# categorical_features = boolean mask for categorical columns
# sparse = False output an array not sparse matrix
df_ohe = ohe.fit_transform(df[categorical_cols]).toarray()
df_ohe

# or 
ohe = OneHotEncoder(sparse=False) 
df_ohe = ohe.fit_transform(df[categorical_cols])
df_ohe

# use all the sorted unique values in categorical columns as new column names
def colNames(df, categorical_cols):
    nms=[]
    for col in categorical_cols:
        nms += [col +'_' + el for el in sorted(df[col].unique())]
    return nms
columns = colNames(df2, categorical_cols)
columns

new = pd.DataFrame(df_ohe, columns =columns)
df3=df2.join(new)
df3
# or
df3 = pd.concat([df2, new], axis=1)
df3



from sklearn.preprocessing import OneHotEncoder

import pandas as pd
data = pd.DataFrame({'Car_Manufacturer' : ['Toyota', 'Ford', 'Ford', 'Mercedes', 'Ford']})

enc = OneHotEncoder( )  #enc = OneHotEncoder(sparse = False) then no need for ...toarrau()
new = enc.fit_transform(data['Car_Manufacturer'].values.reshape(-1, 1)).toarray()
new = pd.DataFrame(new, columns =sorted(data['Car_Manufacturer'].unique()))
new

# or
enc = OneHotEncoder( )  #enc = OneHotEncoder(sparse = False) then no need for ...toarrau()
new = enc.fit_transform(data[['Car_Manufacturer']]).toarray()
new = pd.DataFrame(new, columns =sorted(data['Car_Manufacturer'].unique()))
new


data=data.join(new)
data
data = pd.concat([data, new], axis=1)
data


'''get dummies'''
import pandas as pd
data = pd.DataFrame({'Car_Manufacturer' : ['Toyota', 'Ford', 'Ford', 'Mercedes', 'Ford']})
data
one_hot_encodings = pd.get_dummies(data, columns=['Car_Manufacturer'])
one_hot_encodings
data.join(one_hot_encodings)


'''mapping by dictionary or lambda function'''

import pandas as pd
data = pd.DataFrame({'Exam_Grade' : ['A', 'B', 'A', 'C', 'A']})
data
mapping = {'A' : 5, 'B' : 4, 'C' : 3, 'D' : 2, 'F' : 1}
data['Exam_Grade'] = data['Exam_Grade'].map(mapping)
data
mapping2 = lambda x: 1 if x=='A' else 2 if x=='B'else 0
data['Exam_Grade2'] = data['Exam_Grade'].map(mapping2)
data


'''condition'''
import pandas as pd
numbers = {'nums': [1,2,3,4,5,6,7,8,9,10]}
df = pd.DataFrame(numbers,columns=['nums'])

df.loc[df['nums'] <= 4,'<=4'] = 'True' 
df.loc[df['nums'] > 4, '<=4'] = 'False' 
print (df)

df2 = df[df['set_of_numbers'] <=4]

numbers = {'nums': [1,2,3,4,5,6,7,8,9,10], 'n2': range(10)}
df = pd.DataFrame(numbers,columns=['nums', 'n2'])
df
df.loc[df['nums'] <4, 'n2'] =0


'''DictVectorizer'''
# https://towardsdatascience.com/encoding-categorical-features-21a2651a065c





































































