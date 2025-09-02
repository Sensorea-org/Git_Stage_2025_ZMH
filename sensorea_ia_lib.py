import requests
import pandas as pd
import torch
import numpy as np
import os
import joblib
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import timedelta
from streamlit_echarts import st_echarts
import streamlit as st
import json
from collections.abc import Mapping, Sequence
from graphviz import Digraph

# assistant
#arbre de décision
#on va definir ce que c'est un noeud et comment decider le prochain
class node():
    def __init__(self, type, text, options, on_action=False):
        self.type = type
        self.text = text
        self.options = options
        self.on_action = on_action

    def next(self, model_output, le, tree, labels):

        next_ = []
        for i in range(len(le.classes_)):
            if le.classes_[i] in labels:
                next_.append(model_output[i])
        pred_class = labels[next_.index(max(next_))]
        prob = [float(prob) for prob in next_]
        dic = {"probabilités": prob, "labels": labels}
        confidence = max(next_)
        self.type = tree["nodes"][pred_class]["type"]
        self.text = tree["nodes"][pred_class]["text"]
        self.options = tree["nodes"][pred_class]["options"]
        self.on_action = tree["nodes"][pred_class]["on_action"]
        return pred_class, dic

def render_tree(tree_json, current_id: str):
    dot = Digraph("decision_tree", graph_attr={"rankdir": "LR", "bgcolor": "transparent"})
    dot.attr("node", style="filled,rounded", shape="box", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # styles par type de nœud
    type_style = {
        "message": {"fillcolor": "#E3F2FD", "color": "#1E88E5"},
        "action": {"fillcolor": "#E8F5E9", "color": "#43A047"},
        "form": {"fillcolor": "#FFF3E0", "color": "#FB8C00"},
        "handoff": {"fillcolor": "#FCE4EC", "color": "#D81B60"},
    }

    # nœuds
    for nid, cfg in tree_json["nodes"].items():
        base = type_style.get(cfg["type"], {"fillcolor": "#F5F5F5", "color": "#9E9E9E"})
        fill = base["fillcolor"]
        # surbrillance du nœud courant
        if nid == current_id:
            # teinte un peu plus saturée pour mettre en évidence
            if cfg["type"] == "message":
                fill = "#B3E5FC"
            elif cfg["type"] == "action":
                fill = "#C8E6C9"
            elif cfg["type"] == "form":
                fill = "#FFE0B2"
            else:
                fill = "#F8BBD0"
        label = f"{nid}\\n[{cfg['type']}]"
        dot.node(nid, label=label, fillcolor=fill, color=base["color"])

    # arêtes (avec labels)
    for nid, cfg in tree_json["nodes"].items():
        for opt in cfg.get("options", []):
            target = opt["next"]
            lbl = opt.get("label", "")
            if target not in tree_json["nodes"]:
                # cible manquante : on crée un placeholder
                dot.node(target, label=target, style="dashed,rounded", fillcolor="#EEEEEE", color="#9E9E9E")
            dot.edge(nid, target, label=lbl)

    # nœud "Start" (optionnel, pour montrer le point d'entrée)
    start_id = tree_json.get("start")
    if start_id:
        dot.node("__START__", label="▶ start", shape="ellipse", style="filled", fillcolor="#E0E0E0",
                 color="#616161")
        dot.edge("__START__", start_id, style="dashed")

    return dot
def to_plain(obj):
    if isinstance(obj, Mapping):
        return {k: to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [to_plain(v) for v in obj]
    else:
        return obj
def mk_trend(c, x):
    coeffs = np.polyfit(x, c, deg=1)
    a, b = coeffs
    return a, b
def plot_trend(param,names,trend,bornes):
    names.append("")
    names[1:] = names[0:-1]
    names[0]= "benchmark"
    with open("./data/param.json","r") as f:
        data = json.load(f)
        f.close()
    A_Bench = data[trend]["a"]
    B_Bench = data[trend]["b"]
    axes = data[trend]["label"]
    if axes[1]=="%":
        X = [0,100]
    if axes[1]=="dd":
        X = [0,25]
    X = np.linspace(min(X),max(X),1000)

    trend_dic =[]
    segment_Bench = []
    for i in range(len(X)):
        segment_Bench.append([X[i],A_Bench * X[i] + B_Bench])
    trend_dic.append(segment_Bench)
    for j in range(len(param)):
        trend_actual = []
        a,b = param[j]
        for i in range(len(X)):
            y_i = a*X[i] + b
            x_i = X[i]
            if bornes[j][1]<=x_i<=bornes[j][0]:
                trend_actual.append([x_i,y_i])

        trend_dic.append(trend_actual)

    X = X.tolist()
    options = {
        "title": {
            "text": data[trend]["title"],
        },
        "animationDuration": 1000,
        "tooltip": {"trigger": "axis",
                    "axisPointer": {"type": "shadow"}},
        "xAxis": {
            "type": "value",
            "name": axes[1]
        },
        "yAxis": {"type": "value",
                  "name": axes[0]},
        "series": [{
        "symbolSize": 4,
        "type": "scatter",
        "name":names[k],
        "data": trend_dic[k]
        } for k in range(len(trend_dic))
        ],
    }
    tab1, tab2 = st.tabs(["Chart_actual", "Dataframe"])
    with tab1:
        st_echarts(options=options)
    with tab2:
        for j in range(1,len(names)):
            a, b = param[j-1]
            st.markdown(f"### trend {names[j]}: y = {a:.2f}x + {b:.2f}")
            if b > B_Bench:
                st.markdown(f"* {names[j]} : {data[trend]['instruct'][0]}")
            if b <= B_Bench:
                st.markdown(f"* {names[j]} :{data[trend]['instruct'][1]}")
            if a > A_Bench:
                st.markdown(f"* {names[j]} : {data[trend]['instruct'][2]}")
            if a <= A_Bench:
                st.markdown(f"* {names[j]} : {data[trend]['instruct'][3]}")

#traitement de data

def get_temp(date_datetime,lat =50.8503, lon = 4.3517):
    date = str(date_datetime)
    date = date.split(" ")
    date = date[0]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date}&end_date={date}"
        f"&hourly=temperature_2m"
    )

    response = requests.get(url)
    data = response.json()
    temps = data['hourly']['temperature_2m']
    timestamps = data['hourly']['time']
    df = pd.DataFrame({'time': timestamps, 'temperature': temps})
    df['time'] = pd.to_datetime(df['time'])

    return df['temperature']


def train(root = "."):
        inputs = []
        targets = []
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".pt"):

                        path = dirpath+ "/"+filename
                        print(path)
                        data = torch.load(path,weights_only=False)
                        input_array = data["input"]
                        print(np.shape(input_array))
                        input = input_array.reshape(-1)
                        input = input.numpy()
                        target = data["output"]
                        target = target.numpy()
                        inputs.append(input)
                        targets.append(target)
        X = np.stack(inputs)
        y = np.stack(targets)


        X_train, X_test, y_train_frac, y_test_frac = train_test_split(X, y, test_size=0.2)
        y_train = np.zeros(np.shape(y_train_frac))
        y_test = np.zeros(np.shape(y_test_frac))
        for i in range(len(y_train_frac)):
                for j in range(len(y_train_frac[i])):
                        y_train[i][j] = round(y_train_frac[i][j])
        for i in range(len(y_test_frac)):
                for j in range(len(y_test_frac[i])):
                        y_test[i][j] = round(y_test_frac[i][j])
        base_tree = DecisionTreeClassifier(max_depth=3)
        clf = MultiOutputClassifier(base_tree)
        clf.fit(X_train, y_train)
        joblib.dump(clf, "modele_multioutput_classification.pkl")

        reg = MultiOutputRegressor(
                DecisionTreeRegressor(max_depth=3, min_samples_leaf=20, random_state=0)
        ).fit(X_train, y_train_frac)
        joblib.dump(reg, "modele_multioutput_regression.pkl")

class segment():
    def __init__(self,Cmd,temp_ext,time,water_cons,conso_list,pw):
        self.cmd = Cmd
        self.conso_list = conso_list
        self.temp_ext = temp_ext
        self.time = time
        self.water_cons = water_cons.tolist()
        self.pw = pw
        #self.occupation = occupation
    def gettime(self):

        return self.time
    def getdata(self):

        t = self.time[self.pw].hour
        conso = self.conso_list[self.pw]
        data = self.cmd,conso,self.temp_ext,self.water_cons,t
        return data
    def flatten(self):
        t = self.time[self.pw].hour
        data = []
        for i in self.cmd:
            for j in i:
                data.append(j)
        conso = self.conso_list[self.pw]
        data.append(conso)
        for i in self.temp_ext:
            data.append(i)
        data.append(self.water_cons)
        data.append(t)
        for i in self.conso_list:
            data.append(i)
        return data

def search_sim(input,dataset,threshold):
    sim_ind = []
    list_comp=[1,2,3,4]#élément discriminants pour la similitude de plusieurs situations
    for i in range(len(dataset)):
        a = []
        b = []
        for j in list_comp:
            if type(input.getdata()[j]) == list:
                for k in range(len(input.getdata()[j])):
                    a.append(dataset[i].getdata()[j][k])
                    b.append(input.getdata()[j][k])
            else:
                a.append(dataset[i].getdata()[j])
                b.append(input.getdata()[j])
        a = np.array(a)
        b = np.array(b)

        if np.linalg.norm(a-b)<threshold:
            sim_ind.append(i)
    return sim_ind

def savefile(input,output,pw):
    path = input.gettime()[pw].strftime('%y-%m-%d %H:%M:%S')
    input = input.flatten()
    input = np.array(input)
    if isinstance(input, np.ndarray):
        input_arr = np.array(input, dtype=np.float32)
        input = torch.from_numpy(input_arr)
    if not torch.isnan(input).any():

        output_arr = np.array(output, dtype=np.float32)
        output = torch.from_numpy(output_arr)
        if not torch.isnan(output).any():
            print(output)
            path = path.replace(' ', '')
            path = path.replace('-', '_')
            path = path.replace(':', '_')

            data = {
            "input": input,
            "output": output
            }
            root = "./dataset/newdataset/"
            path = root+path

            # Sauvegarde dans un fichier .pt
            torch.save(data, f"{path}.pt")

            print(f"Fichier '{path}.pt' enregistré avec succès.")
    return path

def readandwrite(dataset,path ="./data/trends.json",pw=19,fw=1):
    with open(path, "r") as f:
        data_loaded = json.load(f)
    dj = data_loaded["dj"]
    temp = data_loaded["temp_ext"]
    data = np.array(data_loaded["commandes"])
    data = list(data.reshape(-1))

    #prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça

    conso_gaz = np.zeros(len(data_loaded["gaz consumption"])-1)
    for i in range(1,len(conso_gaz),1):
        conso_gaz[i]=(data_loaded["gaz consumption"][i]-data_loaded["gaz consumption"][i-1])/10

    conso_water = np.zeros(len(data_loaded["water consumption"])-1)
    for i in range(1,len(conso_water),1):
        conso_water[i]=(data_loaded["water consumption"][i]-data_loaded["water consumption"][i-1])/10


    data.append(conso_gaz[pw])
    for i in temp:
        data.append(i)

    data.append(conso_water[-1])
    t = datetime.today()
    data.append(t.hour)
    for i in conso_gaz[pw+fw:]:
        data.append(i)

    X = np.array(data)
    time_w = []
    delta = timedelta(days=0, hours=0, minutes=15, seconds=0)
    for i in range(pw+fw,0,-1):
        time_w.append(t-i*delta)
    cmd = np.reshape(X[0:240],(pw+fw,12))
    temp_ext = list(X[241:261])
    w_cons = X[261]
    time = X[262]
    gaz_cons = X[263:]
    seg = segment(cmd,temp_ext,time_w,w_cons,gaz_cons,pw)
    sim_ind = search_sim(seg,dataset,20)
    indx = False
    for j in sim_ind:
        min = seg.getdata()[1]
        if dataset[j].getdata()[1]<min:
            min = dataset[j].getdata()[1]
            indx = j
    a = seg.getdata()[0][pw]
    if indx:
        b = dataset[indx].getdata()[0][pw]
    else:
        b = seg.getdata()[0][pw]
    chng = np.array(a)-np.array(b)
    input = seg
    for i in range(len(chng)):
        if a[i]==0:
            chng[i]=0
    path = savefile(input,abs(chng),pw)
    return path
