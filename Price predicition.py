import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Cargamos los datos que van a servir para entrenar  y testear nuestro modelo
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv", index_col = 'Id')

#Diferenciamos columnas categoricas y numericas
numerical_col = df.select_dtypes(exclude ='object')
categorical_col = df.select_dtypes(include = 'object')

#Rellenamos las columnas numericas con entradas vacias con la media del valor de esa columna
for col in numerical_col.columns[numerical_col.isnull().any()]:
    df.fillna(df[col].mean(), inplace = True) 
    df_test.fillna(df_test[col].mean(), inplace = True) 
    
#Rellenamos los espacios con los valores más repetidos en el DF
for col in categorical_col.columns[categorical_col.isnull().any()]:
    df.fillna(df[col].value_counts().index[0] , inplace= True) 
    df_test.fillna(df_test[col].value_counts().index[0], inplace = True) 

# Get dummies(one-hot encoding) para variables categóricas 
aux = pd.get_dummies(df[categorical_col.columns])
aux_test = pd.get_dummies(df_test[categorical_col.columns])

df.drop(columns=categorical_col.columns, axis=1, inplace = True)
df_test.drop(columns=categorical_col.columns, axis=1, inplace = True)

df = pd.concat([aux, df], axis=1)
df_test = pd.concat([aux_test, df_test], axis=1)
df.drop(columns= 'Id', axis=1, inplace = True)

#Si la correlación entre la variable y el precio es superior a 0.5 consideramos que es una variable a tener en cuenta
features = []
for i in df.columns:
    if((abs(df[i].corr(df['SalePrice'])) > 0.5) & (i!='SalePrice')):
        features.append(i)

X = df[features].copy()
y = df.SalePrice
X_test_full = df_test[features].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 0)

models = []
estimators = [100,150,180,200,250]
max_features = np.arange(3,len(features))
max_depth = [None,5,6,7,8,9]
min_samples_leaf = [1,2,3,4]

for i in estimators:
    for j in max_features:
        for k in max_depth:
            for l in min_samples_leaf:
                models.append(RandomForestRegressor(n_estimators=i, max_features=j,max_depth=k, min_samples_leaf=l, random_state=0))

scores = []
best_models = []
contador = 0
def check_models(model, x_t = X_train , x_te = X_test, y_t = y_train, y_te = y_test):
    global contador, best_models
    model.fit(x_t, y_t)
    if((model.score(x_te, y_te)*100) > 86.5):
        print("Entrenamiento: "+ str(model.score(x_t, y_t)*100))
        print("Test " + str(model.score(x_te, y_te)*100))
        contador += 1
        print("Modelo N"+ str(contador))
        print()
        best_models.append(model)

for i in range(0, len(models)):
    check_models(models[i])
    
preds = best_models[117].predict(X_test_full)
output = pd.DataFrame({'Id':X_test_full.index, 'SalePrice': preds} )
output.to_csv('submission.csv', index=False)

#Observación de los parámetros seleccionados por el mejor modelo funcional
#print(best_models[29])
