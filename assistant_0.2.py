import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import json
pw = 19
fw = 1
def degres_heure_glissants(temperatures, t_base=15.0):
    dh_glissant = sum(t_base - t for t in temperatures)
    return dh_glissant
#prediction en arbre
def predict(X,clf):
    input = X
    input = input.reshape(1, -1)
    y_pred = clf.predict(input)
    return y_pred

clf = joblib.load("./modele_multioutput.pkl")
#import fichier json
with open("./data/trends.json", "r") as f:
    data_loaded = json.load(f)


dj = data_loaded["dj"]
temp = data_loaded["temp_ext"]
data = np.array(data_loaded["commandes"])
data = list(data.reshape(-1))

#prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça

conso_gaz = np.zeros(len(data_loaded["gaz consumption"])-1)

for i in range(1,len(conso_gaz),1):
    conso_gaz[i]=data_loaded["gaz consumption"][i]-data_loaded["gaz consumption"][i-1]

conso_water = np.zeros(len(data_loaded["water consumption"])-1)
for i in range(1,len(conso_water),1):
    conso_water[i]=data_loaded["water consumption"][i]-data_loaded["water consumption"][i-1]
t = data_loaded["time"]
data.append(conso_gaz[pw])
for i in temp:
    data.append(i)

data.append(conso_water[-1])
data.append(t)
st.write(len(conso_gaz))
for i in conso_gaz[pw+fw:]:
    data.append(i)

X = np.array(data)
y_pred = predict(X,clf)
feature_names = np.linspace(0,282,283)
class_labels = ["Distribution__Ht_+2_Pmp2A",  "Distribution__Ht_+2_Pmp2B",  "Distribution__Ht_+27_Pmp1A",  "Distribution__Ht_+27_Pmp1B", "Distribution__Ht_-4_Pmp3A","Distribution__Ht_-4_Pmp3B","Distribution__Ht_Rad_Pmp1","Distribution__Ht_Rad_Pmp2","Production_Boiler1","Production_Boiler2","Production_Boiler3","Production_cogen"]#trends

conso_p = np.array(conso_gaz[:pw+fw])

conso = np.array(conso_gaz[pw+fw:])



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

def mk_trend(c,c_p,x,x_p):
    coeffs = np.polyfit(x, c, deg=1)
    a,b = coeffs
    y_inter = np.zeros(len(x))
    #interpolation passsé

    coeffs_ = np.polyfit(x_p,c_p, deg=1)
    a_p,b_p = coeffs_
    y_p = np.zeros(len(x))

    for i in range(len(x)):
        y_inter[i] = a * x[i] + b
        y_p[i] = a_p * x_p[i] + b_p
    return y_inter, y_p,a,b,a_p,b_p
#dico data gaz
y_inter, y_p,ag,bg,a_pg,b_pg = mk_trend(c,c_p,dh,dhp)
d_gaz = pd.DataFrame({
    "x": dh,
    "y (données)": c,
    "y (interpolée)": y_inter,
    "y (past données ) ": c_p,
    "y (past)":y_p
})
#dico data water
conso_p = np.array(conso_water[:pw+fw])
conso = np.array(conso_water[pw+fw:])

occ_data = data_loaded["occupation_list"]
occ_p = occ_data[pw+fw:]
occ = occ_data[:pw+fw]
y_inter, y_p,aw,bw,a_pw,b_pw = mk_trend(conso,conso_p,occ,occ_p)
d_water = pd.DataFrame({
    "x": occ[pw+fw:],
    "y (données)": c,
    "y (interpolée)": y_inter,
    "y (past données ) ": c_p,
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


st.image("./acceuil.jpg")
page1, page2, page3 = st.tabs(["Trends","Prediction","Assistant"])
init = True

with page3:
    with open("./data/assistant.json", "r") as f:
        data_loaded = json.load(f)
    st.write(data_loaded[530:])
with page1:
    st.header("Trends")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("trend : gaz consumption")
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(d_gaz.set_index("x"), height=250)
        tab2.dataframe(d_gaz, height=250, use_container_width=True)
        st.write(f"trend actuelle: y = {ag:.2f}x + {bg:.2f}")
        st.write(f"trend passée: y = {a_pg:.2f}x + {b_pg:.2f}")
        if bg > b_pg:
            st.write("- Possible issue with ECS")
        if bg <= b_pg:
            st.write("- Correct ECS")
        if ag > a_pg:
            st.write("- Possible issue with ventilation or the air temperature")
        if ag <= a_pg:
            st.write("- Correct ventilation and air temperature")

    with col2:
        st.subheader("trend : water consumption")

        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(d_water.set_index("x"), height=250)
        tab2.dataframe(d_water, height=250, use_container_width=True)
        st.write(f"actual trend: y = {aw:.2f}x + {bw:.2f}")
        st.write(f"past trend: y = {a_pw:.2f}x + {b_pw:.2f}")

    with col3:
        st.subheader("trend : electricity consumption")
        tab1, tab2 = st.tabs(["Chart", "Dataframe"])
        tab1.line_chart(d_gaz.set_index("x"), height=250)
        tab2.dataframe(d_gaz, height=250, use_container_width=True)
        st.write(f"trend actuelle: y = {ag:.2f}x + {bg:.2f}")
        st.write(f"trend passée: y = {a_pg:.2f}x + {b_pg:.2f}")
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


