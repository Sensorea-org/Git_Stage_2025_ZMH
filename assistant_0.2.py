import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from graphviz import Digraph
import streamlit_authenticator as stauth
import os, tempfile, shutil, subprocess
from pathlib import Path
from collections.abc import Mapping, Sequence

pw = 19
fw = 1


# --- Convertisseur récursif vers dict/list "plats" ---

def to_plain(obj):
    if isinstance(obj, Mapping):
        return {k: to_plain(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [to_plain(v) for v in obj]
    else:
        return obj

# 1) On récupère les credentials et on les rend mutables/“plats”
credentials = to_plain(st.secrets["credentials"])
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="my_app_cookie",
    cookie_key="random_signature_key",
    cookie_expiry_days=1
)

authenticator.login(location="main")
auth_status = st.session_state.get("authentication_status")
name = st.session_state.get("name")
username = st.session_state.get("username")


def _main_():
    import json
    def degres_heure_glissants(temperatures, t_base=15.0):
        dh_glissant = sum(t_base - t for t in temperatures)
        return dh_glissant

    # prediction en arbre
    def predict_classification(X, clf):
        input = X
        input = input.reshape(1, -1)
        y_pred = clf.predict(input)
        return y_pred

    def predict_reg(X, reg):
        input = X.reshape(1, -1)
        y_pred = reg.predict(input)
        return y_pred

    clf = joblib.load("./modele_multioutput_classification.pkl")
    reg = joblib.load("./modele_multioutput_regression.pkl")
    # import fichier json
    trends_paths = ["THE Hotel", "autre hotel random"]
    name = st.sidebar.radio("Choisir l'hôtel à afficher", trends_paths, key="hotel")
    hotels_paths = ["./data/trends_TH.json", "./data/trends_TH.json"]
    ind = trends_paths.index(name)
    with open(hotels_paths[ind], "r") as f:
        data_loaded = json.load(f)
        data_room = data_loaded["room"]

    dj = data_loaded["dj"]
    temp = data_loaded["temp_ext"]
    data = np.array(data_loaded["commandes"])
    data = list(data.reshape(-1))

    # prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça

    conso_gaz = np.zeros(len(data_loaded["gaz consumption"]) - 1)

    for i in range(1, len(conso_gaz), 1):
        conso_gaz[i] = data_loaded["gaz consumption"][i] - data_loaded["gaz consumption"][i - 1]

    conso_water = np.zeros(len(data_loaded["water consumption"]) - 1)
    for i in range(1, len(conso_water), 1):
        conso_water[i] = data_loaded["water consumption"][i] - data_loaded["water consumption"][i - 1]

    conso_elec = np.zeros(len(data_loaded['electricity consumption']) - 1)
    for i in range(1, len(conso_elec), 1):
        conso_elec[i] = (data_loaded['electricity consumption'][i] - data_loaded['electricity consumption'][
            i - 1]) / 1000

    t = data_loaded["time"]
    data.append(conso_gaz[pw])
    for i in temp:
        data.append(i)

    data.append(conso_water[-1])
    data.append(t)
    for i in conso_gaz[pw + fw:]:
        data.append(i)

    X = np.array(data)
    y_pred_class = predict_classification(X, clf)
    y_pred_regression = predict_reg(X, reg)
    feature_names = np.linspace(0, 342, 343)
    class_labels = ["pompe -4 mod", "pompe +2 mod", "pompe +27 mod", "boilers mod"]
    conso_p = np.array(conso_gaz[:pw + fw])

    conso = np.array(conso_gaz[pw + fw:])

    dh = []
    dhp = []
    c = []
    c_p = []
    w = 20

    # interpolation présent
    c = conso[::4]  # divisé par 4 pcq on a des données toutes les 15 min
    c_p = conso_p[::4]  # divisé par 4 pcq on a des données toutes les 15 min
    dh = dj[int(w / 4):]
    dhp = dj[:int(w / 4)]

    def mk_trend(c, c_p, x, x_p):
        coeffs = np.polyfit(x, c, deg=1)
        a, b = coeffs
        y_inter = np.zeros(len(x))
        # interpolation passsé
        coeffs_ = np.polyfit(x_p, c_p, deg=1)
        a_p, b_p = coeffs_
        y_p = np.zeros(len(x))

        for i in range(len(x)):
            y_inter[i] = a * x[i] + b
            y_p[i] = a_p * x_p[i] + b_p
        return y_inter, y_p, a, b, a_p, b_p

    # dico data gaz
    y_interg, y_pg, ag, bg, a_pg, b_pg = mk_trend(c, c_p, dh, dhp)
    d_gaz = pd.DataFrame({
        "x": dh,
        "y (interpolée)": y_interg,
    })
    d_gaz_p = pd.DataFrame({
        "x": dhp,
        "y (past)": y_pg
    })
    # dico data water

    conso_p = np.array(conso_water[:pw + fw])
    conso = np.array(conso_water[pw + fw:])

    occ_data = data_loaded["occupation_list"]
    occ_p = occ_data[:pw + fw]
    occ = occ_data[pw + fw:]
    y_interw, y_pw, aw, bw, a_pw, b_pw = mk_trend(conso, conso_p, occ, occ_p)
    d_water = pd.DataFrame({
        "x": occ,
        "y (interpolée)": y_interw
    })
    d_water_p = pd.DataFrame({
        "x": occ_p,
        "y (past)": y_pw
    })
    # conso elec
    conso_p = np.array(conso_elec[:pw + fw])
    conso = np.array(conso_elec[pw + fw:])

    y_intere, y_pe, ae, be, a_pe, b_pe = mk_trend(conso, conso_p, occ, occ_p)
    d_elec = pd.DataFrame({
        "x": occ,
        "y (interpolée)": y_intere
    })
    d_elec_p = pd.DataFrame({
        "x": occ_p,
        "y (past)": y_pe
    })
    # création des labels de commandes
    cmd_labels = []
    for i in range(300):
        cmd_labels.append("Data")
    cmd_labels.append("gas consumption")
    for i in range(20):
        cmd_labels.append("temperature")
    cmd_labels.append("water consumption")
    cmd_labels.append("hour")
    for i in range(20):
        cmd_labels.append("gas consumption")

    # LLM implémentation
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

    import json
    def get_temp_chill():
        temp = 2
        return temp

    def get_temp_boiler():
        temp = 2
        return temp

    def GPXX_check():
        temp = 2
        return temp

    def room_check(input,data=data_room):

        words = input.split(" ")
        for word in words:
            try:
                output = int(word)
                output = data[str(output)]
            except:
                output = "room has not been recognized"
        return output

    def get_occupation(data=data_loaded):
        return f"occupation : {data['occupation_list'][-1]}"

    def get_elec_consumption(data=data_loaded):
        return f"consomation elec : {data['electricity consumption'][-1]}"

    def get_gaz_consumption(data=data_loaded):
        return f"consomation gaz : {data['gaz consumption'][-1]}"

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

    tree_json = {
        "start": "acceuil",
        "nodes": {
            "acceuil": {"type": "message",
                        "text": """welcom to the client service, please choose an option : 
                                    """,
                        "options": [{"label": "Hotel_info", "next": "Hotel_info"},
                                    {"label": "fixing", "next": "fixing"}],
                        "on_action": False},
            "Hotel_info": {"type": "message",
                           "text": """welcom to the Informations client service""",
                           "options": [{"label": "home", "next": "acceuil"},
                                       {"label": "consumption", "next": "get_consumption"},
                                       {"label": "occupation", "next": "get_occupation"},
                                       {"label": "rooms", "next": "rooms"},
                                       {"label": "temperature at the ouput off the boiler", "next": "temp_boiler"},
                                       {"label": "temperature at the ouput off the chiller", "next": "temp_chill"},
                                       {"label": "ventilation", "next": "ventilation"}],
                           "on_action": False},
            "fixing": {"type": "message",
                       "text": """welcom to the fixing client service""",
                       "options": [{"label": "home", "next": "acceuil"},
                                   {"label": "chill production", "next": "chill_production"},
                                   {"label": "heat production", "next": "heat_production"}],
                       "on_action": False},
            "ventilation": {"type": "action",
                            "text": "welcom to the ventilation client service",
                            "options": [],
                            "on_action": GPXX_check},
            "rooms": {"type": "form",
                      "text": "welcom to the room client service, please enter the room",
                      "options": [],
                      "on_action": room_check},

            "temp_chill": {"type": "action",
                           "text": False,
                           "options": [],
                           "on_action": get_temp_chill
                           },
            "temp_boiler": {"type": "action",
                            "text": False,
                            "options": [],
                            "on_action": get_temp_boiler
                            },
            "get_consumption": {"type": "message",
                                "text": "welcom to the consumption client service",
                                "options": [{"label": "home", "next": "acceuil"},
                                            {"label": "elec consumption", "next": "get_elec_consumption"},
                                            {"label": "gaz consumption", "next": "get_gaz_consumption"}],
                                "on_action": False},
            "get_elec_consumption": {"type": "action",
                                     "text": "welcom to the elec consumption client service",
                                     "options": [],
                                     "on_action": get_elec_consumption},
            "get_gaz_consumption": {"type": "action",
                                    "text": "welcom to the gaz consumption client service",
                                    "options": [],
                                    "on_action": get_gaz_consumption},
            "get_occupation": {"type": "action",
                               "text": "welcom to the occupation client service",
                               "options": [],
                               "on_action": get_occupation},
            "heat_production": {"type": "message",
                                "text": "welcom to the heat production client service",
                                "options": [{"label": "home", "next": "acceuil"},
                                            {"label": "boiler problem", "next": "boiler_fixing"},
                                            {"label": "cogen problem", "next": "cogen_fixing"}],
                                "on_action": False},
            "boiler_fixing": {"type": "message",
                              "text": """welcom to the boiler client service

                      •	Température de départ est trop basse : regarder le tableau des automates (disjoncteurs ou thermique),
                       vérifier la pression, vérifier la distribution pour s’assurer qu’il y a du débit (mais si T basse -> distribution fonctionne normalement), 
                       aller voir sur place, vérifier le gaz avec la société de maintenance


                        o	Solution : forcer temporairement une ou plusieurs chaudières en manuel et sur la GTC
    """,
                              "options": [{"label": "home", "next": "acceuil"}],
                              "on_action": False},
            "cogen_fixing": {"type": "message",
                             "text": """welcom to the cogen client service
                      Pas assez de demande  les ballons sont trop chauds  elle ne démarre plus

    """,
                             "options": [{"label": "home", "next": "acceuil"}],
                             "on_action": False},
            "chill_production": {"type": "message",
                                 "text": "welcom to the chill production client service",
                                 "options": [{"label": "home", "next": "acceuil"},
                                             {"label": "chiller problem", "next": "chiller_fixing"}],
                                 "on_action": False},
            "chiller_fixing": {"type": "message",
                               "text": """welcom to the chiller client service
                     Les tours n’ont pas assez refroidi (message : pression trop haute)


                     Essayer de maintenir 28°C

    """,
                               "options": [{"label": "home", "next": "acceuil"}],
                               "on_action": False},

        }
    }

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    clf_T = joblib.load("modele_LLM_SVC.pkl")
    clf_tfidf = joblib.load("modele_LLM_tfidf.pkl")

    def prediction_T(input, model, clf_T):
        X_emb = model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
        proba = clf_T.predict_proba(X_emb)[0]
        return proba

    le = LabelEncoder()
    y = le.fit_transform(['Hotel_info', 'acceuil', 'boiler_fixing', 'chill_production',
                          'chiller_fixing', 'cogen_fixing', 'fixing', 'get_consumption',
                          'get_elec_consumption', 'get_gaz_consumption', 'get_occupation',
                          'heat_production', 'rooms', 'temp_boiler', 'temp_chill', 'ventilation'])

    page1, page2, page3 = st.tabs(["Trends", "Prediction", "Assistant"])

    with page3:
        if "node_id" not in st.session_state:
            st.session_state.node_id = tree_json["start"]
        init_node = st.session_state.node_id
        init_type = tree_json["nodes"][init_node]["type"]
        init_text = tree_json["nodes"][init_node]["text"]
        init_options = tree_json["nodes"][init_node]["options"]
        init_on_action = tree_json["nodes"][init_node]["on_action"]
        main_node = node(init_type, init_text, init_options, on_action=init_on_action)
        # Initialize chat history

        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": main_node.text}]
        # Display chat messages from history on app rerun

        # Accept user input

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message == st.session_state.messages[-1]:
                    dot = render_tree(tree_json, st.session_state["node_id"])
                    st.graphviz_chart(dot, use_container_width=True)

        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                request = []
                request.append(prompt)
                proba = prediction_T(request, model, clf_T)
                last_node = st.session_state.node_id
                pred_node, dic = main_node.next(proba, le, tree_json, [opt["next"] for opt in main_node.options])
                st.session_state.node_id = pred_node
                if main_node.type == "message":
                    assistant_response = main_node.text
                    st.write(f"""classe reconnue : {pred_node} avec une proba de {max(dic["probabilités"])}""")
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.rerun()
                if main_node.type == "action":
                    assistant_response = main_node.on_action()
                    st.write(f"""classe reconnue : {pred_node} avec une proba de {max(dic["probabilités"])}""")
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.session_state.node_id = last_node
                    st.rerun()
                if main_node.type == "form":
                    assistant_response = main_node.on_action(prompt)
                    st.write(f"""classe reconnue : {pred_node} avec une proba de {max(dic["probabilités"])}""")
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    st.session_state.node_id = last_node
                    st.rerun()

                if main_node.type == "handoff":
                    assistant_response = main_node.text
                message_placeholder.markdown(assistant_response)
            # Add assistant response to chat history
    with page1:
        st.header("Trends")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("trend : gaz consumption")
            tab1, tab2, tab3 = st.tabs(["Chart_actual", "Chart_past", "Dataframe"])
            tab1.line_chart(d_gaz.set_index("x"), height=250)
            tab2.line_chart(d_gaz_p.set_index("x"), height=250)
            tab3.dataframe(d_gaz, height=250, use_container_width=True)
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

            tab1, tab2, tab3 = st.tabs(["Chart_actual", "Chart_past", "Dataframe"])
            tab1.line_chart(d_water.set_index("x"), height=250)
            tab2.line_chart(d_water_p.set_index("x"), height=250)
            tab3.dataframe(d_water, height=250, use_container_width=True)
            st.write(f"actual trend: y = {aw:.2f}x + {bw:.2f}")
            st.write(f"past trend: y = {a_pw:.2f}x + {b_pw:.2f}")
            if bw > b_pw:
                st.write("- constant use of water increased -> possible leak")
            if bw <= b_pw:
                st.write("- constant use of water decreased 👌")
            if aw > a_pw:
                st.write("- normalized consumption increased")
            if aw <= a_pw:
                st.write("- normalized consumption decreased")

        with col3:

            st.subheader("trend : electricity consumption")
            tab1, tab2, tab3 = st.tabs(["Chart_actual", "Chart_past", "Dataframe"])
            tab1.line_chart(d_elec.set_index("x"), height=250)
            tab2.line_chart(d_elec_p.set_index("x"), height=250)
            tab3.dataframe(d_elec, height=250, use_container_width=True)
            st.write(f"trend actuelle: y = {ae:.2f}x + {be:.2f}")
            st.write(f"trend passée: y = {a_pe:.2f}x + {b_pe:.2f}")
            if be > b_pe:
                st.write("- constant use of electricity increased")
            if be <= b_pe:
                st.write("- constant use of electricity decreased 👌")
            if ae > a_pe:
                st.write("- normalized consumption increased")
            if ae <= a_pe:
                st.write("- normalized consumption decreased")
    with page2:
        st.header("Decision Trees")
        st.header("Results")
        pred_cmd = []
        pred_freq = []
        for i in range(len(class_labels)):
            pred_cmd.append(y_pred_class[0][i])
            pred_freq.append(y_pred_regression[0][i])
        cmd_act = X[296:300]

        dic = pd.DataFrame({
            "command": class_labels,
            "cmd actual": cmd_act,
            "advised cmd": pred_cmd,
            "advised frequency": pred_freq,
        })
        st.dataframe(dic, height=250, use_container_width=True)

        name = st.radio("Choisir le label à afficher", class_labels)
        label_index = class_labels.index(name)
        # Chemin de décision
        tree = reg.estimators_[label_index]
        feat = list(tree.tree_.feature)
        while -2 in feat:
            feat.remove(-2)

        fig, ax = plt.subplots(figsize=(15, 6))
        plot_tree(
            reg.estimators_[label_index],
            feature_names=feature_names,
            class_names=["off", name],
            filled=True,
            rounded=True,
            fontsize=8,
            ax=ax
        )
        st.pyplot(fig)

        feature_ex = []
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


if auth_status is True:
    st.success(f"Bienvenue {name} ({username})")
    authenticator.logout(button_name="Déconnexion", location="sidebar", key="logout_btn")
    _main_()
elif auth_status is False:
    st.error("Identifiants invalides")
else:
    st.info("Veuillez vous connecter")


