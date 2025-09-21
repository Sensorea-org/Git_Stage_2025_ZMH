import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

import streamlit_authenticator as stauth
import os, tempfile, shutil, subprocess
from pathlib import Path
from streamlit_quill import st_quill


from sensorea_ia_lib import *
pw = 19
fw = 1
#login
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
#code main
def _main_():
    import json

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
    with open("./data/trends_TH.json", "r") as f:
        data_loaded = json.load(f)
    data_room = data_loaded["rooms"]

    temp = data_loaded["temp_ext"][-20:]
    data = np.array(data_loaded["commandes"])
    data = list(data.reshape(-1))

    # prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça

    conso_gaz = data_loaded["gaz consumption"]
    conso_water = data_loaded["water consumption"]
    conso_elec = data_loaded['electricity consumption']


    t = data_loaded["time"]
    data.append(conso_gaz[pw])
    for i in temp:
        data.append(i)

    data.append(conso_water[-1])
    data.append(t)
    cons = [(conso_gaz[i]-conso_gaz[i-1]) for i in range(len(conso_gaz)-(pw+fw),len(conso_gaz),1)]
    for i in cons:
        data.append(i)

    X = np.array(data)
    y_pred_class = predict_classification(X, clf)
    y_pred_regression = predict_reg(X, reg)
    feature_names = np.linspace(0, 342, 343)
    class_labels = ["pompe -4 mod", "pompe +2 mod", "pompe +27 mod", "boilers mod"]


    # interpolation présent
    # dico data gaz
    #calcul des dd et des cooling dd


    conso = conso_gaz[::4]
    dd = []
    cdd = []
    l = 24
    for i in range(len(data_loaded["temp_ext"])-l):
        dd.append(degres_heure_glissants(data_loaded["temp_ext"][i:i+l], 15, l))
        inv_temp = [-t for t in data_loaded["temp_ext"][i:i+l]]
        cdd.append(degres_heure_glissants(inv_temp, -18, l))
    c = [(conso[i]-conso[i-l])*(2) for i in range(l,len(conso),1)]
    ddp = dd[-(2*l):-l]
    dd = dd[-l:]
    cddp = cdd[-(2*l):-l]
    cdd = cdd[-l:]
    bcg = [[max(dd), min(dd)],[max(ddp), min(ddp)]]
    cp = c[-(2*l):-l]
    c = c[-l:]

    try:
        ag,bg = mk_trend(c,dd)


    except:
        ag=0
        bg=float(np.mean(c))
    try:
        agp,bgp = mk_trend(cp,ddp)


    except:
        agp=0
        bgp=float(np.mean(cp))

    # dico data water
    conso = np.array(conso_water)[::4]
    c = [(conso[i] - conso[i - l])/((10)/(2)) for i in range(l, len(conso), 1)]

    cp = c[-(2 * l):-l]
    c = c[-l:]

    occ_data = data_loaded["occupation_list"][::4]
    occ_data_p = occ_data[-(2 * l):-l]
    occ_data = occ_data[-l:]

    occ_true_data = data_loaded["True_occupation_list"][::4]
    occ_true_data_p = occ_true_data[-(2 * l):-l]
    occ_true_data = occ_true_data[-l:]

    aw, bw = mk_trend(c,occ_data)
    awt, bwt = mk_trend(c,occ_true_data)
    awp, bwp = mk_trend(cp, occ_data_p)
    awtp, bwtp = mk_trend(cp, occ_true_data_p)

    bcw = [[max(occ_data), min(occ_data)],[max(occ_true_data), min(occ_true_data)],[max(occ_data_p), min(occ_data_p)],[max(occ_true_data_p), min(occ_true_data_p)]]

    # conso elec
    conso = np.array(conso_elec)[::4]
    c = [(conso[i] - conso[i - l])/((1000)/(2)) for i in range(l, len(conso), 1)]
    cp = c[-(2 * l):-l]
    c = c[-l:]
    bce = bcw
    ae, be = mk_trend(c,occ_data)
    aet,bet = mk_trend(c,occ_true_data)
    aep, bep = mk_trend(cp,occ_data_p)
    aetp,betp = mk_trend(cp,occ_true_data_p)

    try:
        aedd,bedd = mk_trend(c,dd)

    except:
        aedd=0
        bedd=float(np.mean(c))

    try:
        aeddp, beddp = mk_trend(cp, ddp)

    except:
        aeddp = 0
        beddp = float(np.mean(cp))

    try:
        aecdd,becdd = mk_trend(c,cdd)

    except:
        aecdd=0
        becdd=float(np.mean(c))

    try:
        aecddp,becddp = mk_trend(cp,cddp)

    except:
        aecddp=0
        becddp=float(np.mean(cp))
    bcecdd = [[max(cdd),min(cdd)],[max(cddp),min(cddp)]]


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

    tree_json = {
        "start": "fixing",
        "nodes": {
            "acceuil": {"type": "message",
                        "text": """## welcome to the client service, please choose an option:
                                        * infos sur l'hotel
                                        * fixing """,
                        "options": [{"label": "Hotel info", "next": "Hotel_info"},
                                    {"label": "fixing", "next": "fixing"}],
                        "on_action": False},
            "Hotel_info": {"type": "message",
                           "text": """## welcom to the Informations client service, please choose an option:
                                        * consommation 
                                        * occupation
                                        * rooms
                                        * temperature at the ouput off the boiler
                                        * temperature at the ouput off the chiller
                                        * ventilation""",
                           "options": [{"label": "home", "next": "acceuil"},
                                       {"label": "consumption", "next": "get_consumption"},
                                       {"label": "occupation", "next": "get_occupation"},
                                       {"label": "rooms", "next": "rooms"},
                                       {"label": "temperature at the ouput off the boiler", "next": "temp_boiler"},
                                       {"label": "temperature at the ouput off the chiller", "next": "temp_chill"},
                                       {"label": "ventilation", "next": "ventilation"}],
                           "on_action": False},
            "fixing": {"type": "message",
                       "text": """## welcom to the fixing client service, please choose an option:
                       * home
                       * chill production
                       * heat production
                       """,
                       "options": [{"label": "home", "next": "acceuil"},
                                   {"label": "chill production", "next": "chill_production"},
                                   {"label": "heat production", "next": "heat_production"}],
                       "on_action": False},
            "ventilation": {"type": "action",
                            "text": "## welcom to the ventilation client service",
                            "options": [],
                            "on_action": GPXX_check},
            "rooms": {"type": "form",
                      "text": "## welcom to the room client service, please enter the room",
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
                                "text": """## welcom to the consumption client service, please choose an option:
                                        * home
                                        * elec consumption
                                        * gaz consumption""",
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
                                "text": """## welcom to the heat production client service, please choose an option:
                                * home
                                * boiler problem
                                * cogen problem""",
                                "options": [{"label": "home", "next": "acceuil"},
                                            {"label": "boiler problem", "next": "boiler_fixing"},
                                            {"label": "cogen problem", "next": "cogen_fixing"}],
                                "on_action": False},
            "boiler_fixing": {"type": "message",
                              "text": """## welcom to the boiler client service

                          *	Température de départ est trop basse : regarder le tableau des automates (disjoncteurs ou thermique),
                           vérifier la pression, vérifier la distribution pour s’assurer qu’il y a du débit (mais si T basse -> distribution fonctionne normalement),
                           aller voir sur place, vérifier le gaz avec la société de maintenance


                          * Solution : forcer temporairement une ou plusieurs chaudières en manuel et sur la GTC
        """,
                              "options": [{"label": "home", "next": "acceuil"}],
                              "on_action": False},
            "cogen_fixing": {"type": "message",
                             "text": """## welcom to the cogen client service
                          * Pas assez de demande  les ballons sont trop chauds  elle ne démarre plus

        """,
                             "options": [{"label": "home", "next": "acceuil"}],
                             "on_action": False},
            "chill_production": {"type": "message",
                                 "text": """## welcom to the chill production client service, please choose an option:
                                 * home
                                 * chiller problem""",
                                 "options": [{"label": "home", "next": "acceuil"},
                                             {"label": "chiller problem", "next": "chiller_fixing"}],
                                 "on_action": False},
            "chiller_fixing": {"type": "message",
                               "text": """## welcom to the chiller client service
                         * Les tours n’ont pas assez refroidi (message : pression trop haute)


                         * Essayer de maintenir 28°C

        """,
                               "options": [{"label": "home", "next": "acceuil"}],
                               "on_action": False},

        }
    }

    model = SentenceTransformer("models/paraphrase-multilingual-mpnet-base-v2")
    clf_T = joblib.load("modele_LLM_SVC.pkl")

    def prediction_T(input, model, clf_T):
        X_emb = model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
        proba = clf_T.predict_proba(X_emb)[0]
        return proba

    le = LabelEncoder()
    y = le.fit_transform(['Hotel_info', 'acceuil', 'boiler_fixing', 'chill_production',
                          'chiller_fixing', 'cogen_fixing','fixing', 'get_consumption',
                          'get_elec_consumption', 'get_gaz_consumption', 'get_occupation',
                          'heat_production', 'rooms', 'temp_boiler', 'temp_chill', 'ventilation'])

    def sensorea_login():
        page1, page2, page3,page4 = st.tabs(["Trends", "Prediction", "Assistant","deboguage"])
        with page1:
            st.header("Trends")
            st.subheader("trend : gaz consumption")
            plot_trend([(ag, bg),(agp,bgp)], ["actual","past"], "gaz", bcg)
            st.subheader("trend : water consumption")
            plot_trend([(aw, bw), (awt, bwt),(awp, bwp), (awtp, bwtp)], ["actual reservation", "actual true occ","past reservation", "past true occ"], "water", bcw)

            st.subheader("trend : electricity consumption")
            plot_trend([(ae, be), (aet, bet),(aep, bep), (aetp, betp)], ["actual reservation", "actual true occ","past reservation", "past true occ"], "elec", bce)

            plot_trend([(aedd,bedd),(aeddp,beddp)], ["actual","past"], "elec_dd", bcg)
            plot_trend([(aecdd,becdd),(aecddp,becddp)], ["actual","past"], "elec_cdd", bcecdd)
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
                    options = [opt["next"] for opt in main_node.options]

                    last_node = st.session_state.node_id
                    pred_node, dic = main_node.next(proba, le, tree_json, options)
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
        with page4:
            st.header("Debogage")

            with open("./data/param.json", "r",encoding="utf-8") as f:
                data = json.load(f)
                f.close()



            content = json.dumps(data, ensure_ascii=False, indent=2)
            edited = st.text_area("Paramètres (JSON)", value=content, height=400)

            d = json.loads(edited)
            st.write("veillez à clear le cache lors d'une mise à jour des fichiers")
            if st.button("update"):
                with open("./data/param.json", "w+") as f:
                    json.dump(d, f)
                    f.close()
                st.write("update correctly pushed")
                st.cache_data.clear()
                st.rerun()
    def technician_login():
        page1, page2, page3 = st.tabs(["Trends", "Prediction", "Assistant"])


        with page1:
            st.header("Trends")
            st.subheader("trend : gaz consumption")
            plot_trend([(ag, bg)], ["actual"], "gaz", bcg)
            st.subheader("trend : water consumption")
            plot_trend([(aw, bw), (awt, bwt)], ["actual reservation", "actual true occ"], "water", bcw)

            st.subheader("trend : electricity consumption")
            plot_trend([(ae, be), (aet, bet)], ["actual", "actual true occ"], "elec", bce)

            plot_trend([(aedd,bedd)], ["acctual"], "elec_dd", bcg)
            plot_trend([(aecdd,becdd)], ["actual"], "elec_cdd", bcecdd)
            # fonction toute précooked pour rajouter une courbe il suffit de rajouter le nome et les paramètres dans la liste exemple:
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
                    st.markdown(message["content"])

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
                    options =[opt["next"] for opt in main_node.options]
                    last_node = st.session_state.node_id
                    pred_node, dic = main_node.next(proba, le, tree_json, options)
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
    if auth_status is True:
        name = st.session_state.get("name")
        if name == "sensorea":
            tree_json["start"]="acceuil"
            sensorea_login()
        if name != "sensorea":
            tree_json["start"] = "fixing"
            technician_login()
if auth_status is True:
    st.success(f"Bienvenue {name} ({username})")
    authenticator.logout(button_name="Déconnexion", location="sidebar", key="logout_btn")
    _main_()
elif auth_status is False:
    st.error("Identifiants invalides")
else:
    st.info("Veuillez vous connecter")