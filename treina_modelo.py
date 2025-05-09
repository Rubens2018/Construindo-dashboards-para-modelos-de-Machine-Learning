from ucimlrepo import fetch_ucirepo

heart_disease = fetch_ucirepo(id=45)
dados = heart_disease.data.features
dados["doenca"] = (heart_disease.data.targets > 0) * 1

x = dados.drop(columns=["doenca"])
y = dados["doenca"]

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=432, stratify=y)

import xgboost as xgb
modelo = xgb.XGBClassifier(objective="binary:logistic")
modelo.fit(x_treino, y_treino)
preds = modelo.predict(x_teste)

from sklearn.metrics import accuracy_score
acuracia = accuracy_score(y_teste, preds)
print(f"Acurácia: {acuracia:.2f}")

import joblib
joblib.dump(modelo, "modelo_xgboost.pkl")