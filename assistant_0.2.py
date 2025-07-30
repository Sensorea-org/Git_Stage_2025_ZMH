import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime

def degres_heure_glissants(temperatures, t_base=15.0):
    dh_glissant = sum(t_base - t for t in temperatures)
    return dh_glissant
#prediction en arbre
def predict(X,clf):
    input = X
    input = input.reshape(1, -1)
    y_pred = clf.predict(input)
    return y_pred

clf = joblib.load("C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/modele_multioutput.pkl")
#import data
test = torch.load('C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/temp.pt',weights_only=False)
test_past = torch.load('C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/temp_past.pt',weights_only=False)
test_past_past = torch.load('C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/temp_past_past.pt',weights_only=False)
#import fichier json
with open("C:/Users\hugom\OneDrive\Documents\Stage_2025\data/trends.json", "r") as f:
    data_loaded = json.load(f)
trends = {'occupation_list':data_loaded['occupation_list'],
          'water consumption':data_loaded['water consumption'],
          'electricity consumption':data_loaded['electricity consumption']}
dj = data_loaded["dj"]
temp = data_loaded["temp_ext"]
data = np.array(data_loaded["commandes"])
data = list(data.reshape(-1))
#prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça
X = test['input']
X = X.reshape(-1)
X = X.numpy()
conso = X[240]
X = X[261:]
data.append(conso)
for i in temp:
    data.append(i)
for i in X:
    data.append(i)
X = np.array(data)

y_pred = predict(X,clf)
feature_names = np.linspace(0,282,283)
class_labels = ["Distribution__Ht_+2_Pmp2A",  "Distribution__Ht_+2_Pmp2B",  "Distribution__Ht_+27_Pmp1A",  "Distribution__Ht_+27_Pmp1B", "Distribution__Ht_-4_Pmp3A","Distribution__Ht_-4_Pmp3B","Distribution__Ht_Rad_Pmp1","Distribution__Ht_Rad_Pmp2","Production_Boiler1","Production_Boiler2","Production_Boiler3","Production_cogen"]#trends

conso = np.array([test['input'][263:283]])[0]

conso_p = np.array([test_past['input'][263:283]])[0]



dh = []
dhp = []
c = []
c_p = []
w = 20

#interpolation présent
c = conso[::4]#divisé par 4 pcq on a des données toutes les 15 min
c_p = conso_p[::4]#divisé par 4 pcq on a des données toutes les 15 min
dh = dj[int(w/4):]
dhp = dj[:int(w/4)]
coeffs = np.polyfit(dh, c, deg=1)
a,b = coeffs
y_inter = np.zeros(len(dh))
#interpolation passsé

coeffs_ = np.polyfit(dhp,c_p, deg=1)
a_p,b_p = coeffs_
y_p = np.zeros(len(dh))

for i in range(len(dh)):
    y_inter[i] = a * dh[i] + b
    y_p[i] = a_p * dhp[i] + b_p

df = pd.DataFrame({
    "x": dh,
    "y (données)": c,
    "y (interpolée)": y_inter,
    "y (past)":y_p
})

#création des labels de commandes
cmd_labels = []
for i in range(240):
    cmd_labels.append("command")
cmd_labels.append("gas consumption")
for i in range(20):
    cmd_labels.append("temperature")
cmd_labels.append("water consumption")
cmd_labels.append("hour")
for i in range(20):
    cmd_labels.append("gas consumption")


st.image("C:/Users/hugom/OneDrive/Documents/Stage_2025/img.jpg")
page1, page2, page3 = st.tabs(["Trends","Prediction","Assistant"])
init = True

with page3:
    with open("C:/Users\hugom\OneDrive\Documents\Stage_2025\data/assistant.json", "r") as f:
        data_loaded = json.load(f)
    st.write(data_loaded[530:])
with page1:
    st.header("Trends")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("trend : gaz consumption")
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(df.set_index("x"), height=250)
        tab2.dataframe(df, height=250, use_container_width=True)
        st.write(f"trend actuelle: y = {a:.2f}x + {b:.2f}")
        st.write(f"trend passée: y = {a_p:.2f}x + {b_p:.2f}")
        if b > b_p:
            st.write("- Possible issue with ECS")
        if b <= b_p:
            st.write("- Correct ECS")
        if a > a_p:
            st.write("- Possible issue with ventilation or the air temperature")
        if a <= a_p:
            st.write("- Correct ventilation and air temperature")

    with col2:
        st.subheader("trend : water consumption")

        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(df.set_index("x"), height=250)
        tab2.dataframe(trends, height=250, use_container_width=True)
        st.write(f"actual trend: y = {a:.2f}x + {b:.2f}")
        st.write(f"past trend: y = {a_p:.2f}x + {b_p:.2f}")

    with col3:
        st.subheader("trend : electricity consumption")
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(df.set_index("x"), height=250)
        tab2.dataframe(df, height=250, use_container_width=True)
        st.write(f"trend actuelle: y = {a:.2f}x + {b:.2f}")
        st.write(f"trend passée: y = {a_p:.2f}x + {b_p:.2f}")
with page2:
    st.header("Decision Trees")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Results")
        for i in range(len(class_labels)):
            if y_pred[0][i] == 1:
                if i == len(class_labels) - 1:
                    st.write(f"- alright no probs (reference to the model 😹😹)")
                else:
                    st.write(f"- 🌡️ dérive de commande pour la {class_labels[i]} ")
    with col2:
        cmd_act = X[228:240]
        dic = pd.DataFrame({
            "command": class_labels,
            "cmd actual": cmd_act,
        })
        st.dataframe(dic, height=250, use_container_width=True)

    label_index = st.slider("Choisir le label à afficher", 0, len(clf.estimators_) - 1)
    # Chemin de décision
    tree = clf.estimators_[label_index]
    feat = list(tree.tree_.feature)
    while -2 in feat:
        feat.remove(-2)


    fig, ax = plt.subplots(figsize=(12, 6))
    name = class_labels[label_index]
    plot_tree(
        clf.estimators_[label_index],
        feature_names=feature_names,
        class_names=["off", name],
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax
    )
    st.pyplot(fig)


    feature_ex=[]
    inp_ex = []
    label_ex = []
    for i in feat:
        feature_ex.append(feature_names[i])
        inp_ex.append(X[i])
        label_ex.append(cmd_labels[i])
    c_ex = pd.DataFrame({
            "command index": feature_ex,
            "cmd actual": inp_ex,
            "cmd label": label_ex
        })
    st.dataframe(c_ex, height=250, use_container_width=True)


