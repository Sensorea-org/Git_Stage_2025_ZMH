import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import joblib
from graphviz import Digraph
def render_tree(tree_json, current_id: str):
    dot = Digraph("decision_tree", graph_attr={"rankdir": "LR", "bgcolor": "transparent"})
    dot.attr("node", style="filled,rounded", shape="box", fontname="Helvetica", fontsize="10")
    dot.attr("edge", fontname="Helvetica", fontsize="9")

    # styles par type de nœud
    type_style = {
        "message": {"fillcolor": "#E3F2FD", "color": "#1E88E5"},
        "action":  {"fillcolor": "#E8F5E9", "color": "#43A047"},
        "form":    {"fillcolor": "#FFF3E0", "color": "#FB8C00"},
        "handoff": {"fillcolor": "#FCE4EC", "color": "#D81B60"},
    }

    # nœuds
    for nid, cfg in tree_json["nodes"].items():
        base = type_style.get(cfg["type"], {"fillcolor": "#F5F5F5", "color": "#9E9E9E"})
        fill = base["fillcolor"]
        # surbrillance du nœud courant
        if nid == current_id:
            # teinte un peu plus saturée pour mettre en évidence
            if cfg["type"] == "message": fill = "#B3E5FC"
            elif cfg["type"] == "action": fill = "#C8E6C9"
            elif cfg["type"] == "form": fill = "#FFE0B2"
            else: fill = "#F8BBD0"
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
        dot.node("__START__", label="▶ start", shape="ellipse", style="filled", fillcolor="#E0E0E0", color="#616161")
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
def room_check():
    temp = 2
    return temp
def get_occupation():
    temp = 2
    return temp
def get_elec_consumption():
    temp = 2
    return temp
def get_gaz_consumption():
    temp = 2
    return temp

class node():
    def __init__(self,type,text,options,on_action=False):
        self.type=type
        self.text=text
        self.options =options
        self.on_action=on_action
    def next(self,model_output,le,tree,labels):

        next_ = []
        for i in range(len(le.classes_)):
            if le.classes_[i] in labels:
                next_.append(model_output[i])
        pred_class = labels[next_.index(max(next_))]
        prob = [float(prob) for prob in next_]
        dic ={"probabilités":prob,"labels":labels}
        confidence = max(next_)
        self.type=tree["nodes"][pred_class]["type"]
        self.text= tree["nodes"][pred_class]["text"]
        self.options = tree["nodes"][pred_class]["options"]
        self.on_action=tree["nodes"][pred_class]["on_action"]
        return pred_class,dic
tree_json = {
    "start":"acceuil",
    "nodes":{
        "acceuil":{"type":"message",
                               "text":"""welcom to the client service, please choose an option : 
                                """,
                               "options":[{"label":"Hotel_info","next":"Hotel_info"},
                                          {"label":"fixing","next":"fixing"}],
                               "on_action":False},
        "Hotel_info":{"type":"message",
                                "text":"""welcom to the Informations client service""",
                                "options":[{"label":"chiller/boiler room","next":"chiller/boiler_room"},
                                                {"label":"ventilation","next":"ventilation"},
                                                {"label":"rooms","next":"rooms"},
                                          {"label":"home","next":"acceuil"},
                                          {"label":"consumption","next":"get_consumption"},
                                          {"label":"occupation","next":"get_occupation"}],
                                "on_action":False},
        "fixing":{"type":"message",
                                "text":"""welcom to the fixing client service""",
                                "options":[{"label":"home","next":"acceuil"},
                                            {"label":"chill production","next":"chill_production"},
                                           {"label":"heat production","next":"heat_production"}],
                                "on_action":False},

        "chiller/boiler_room":{"type":"message",
                               "text":"""welcom to the chiller and boiler client service,make a choice:""",
                               "options":[{"label":"temperature at the ouput off the chiller","next":"temp_chill"},{"label":"temperature at the ouput off the boiler","next":"temp_boiler"},
                                          {"label":"home","next":"acceuil"}],
                               "on_action":False},
        "ventilation":{"type":"action",
                       "text":"welcom to the ventilation client service",
                       "options":[{"label":"home","next":"acceuil"}],
                       "on_action":GPXX_check},
        "rooms":{"type":"form",
                 "text":"welcom to the room client service, please enter the room",
                 "options":[{"label":"home","next":"acceuil"}],
                 "on_action":room_check},

        "temp_chill":{"type":"action",
                              "text":False,
                              "options":[{"label":"home","next":"acceuil"}],
                              "on_action":get_temp_chill
                              },
        "temp_boiler":{"type":"action",
                              "text":False,
                              "options":[{"label":"home","next":"acceuil"}],
                              "on_action":get_temp_boiler
                              },
        "get_consumption":{"type":"message",
                           "text":"welcom to the consumption client service",
                           "options":[{"label":"gaz consumption","next":"get_gaz_consumption"},
                                    {"label":"elec consumption","next":"get_elec_consumption"},
                                      {"label":"home","next":"acceuil"}],
                           "on_action":False},
        "get_elec_consumption":{"type":"action",
                           "text":"welcom to the elec consumption client service",
                           "options":[{"label":"home","next":"acceuil"}],
                           "on_action":get_elec_consumption},
        "get_gaz_consumption":{"type":"action",
                           "text":"welcom to the gaz consumption client service",
                           "options":[{"label":"home","next":"acceuil"}],
                           "on_action":get_gaz_consumption},
        "get_occupation":{"type":"action",
                           "text":"welcom to the occupation client service",
                           "options":[{"label":"home","next":"acceuil"}],
                           "on_action":get_occupation},
        "heat_production":{"type":"message",
                               "text":"welcom to the heat production client service",
                           "options":[{"label":"home","next":"acceuil"},
                                      {"label":"boiler problem","next":"boiler_fixing"},
                                      {"label":"cogen problem","next":"cogen_fixing"}],
                           "on_action":False},
        "boiler_fixing":{"type":"message",
                  "text":"""welcom to the boiler client service
                  
                  •	Température de départ est trop basse : regarder le tableau des automates (disjoncteurs ou thermique),
                   vérifier la pression, vérifier la distribution pour s’assurer qu’il y a du débit (mais si T basse -> distribution fonctionne normalement), 
                   aller voir sur place, vérifier le gaz avec la société de maintenance


                    o	Solution : forcer temporairement une ou plusieurs chaudières en manuel et sur la GTC
""",
                  "options":[{"label":"home","next":"acceuil"}],
                  "on_action":False},
        "cogen_fixing":{"type":"message",
                  "text":"""welcom to the cogen client service
                  Pas assez de demande  les ballons sont trop chauds  elle ne démarre plus
                  
""",
                  "options":[{"label":"home","next":"acceuil"}],
                  "on_action":False},
        "chill_production":{"type":"message",
                               "text":"welcom to the heat production client service",
                           "options":[{"label":"home","next":"acceuil"},
                                      {"label":"chiller problem","next":"chiller_fixing"}],
                           "on_action":False},
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


def prediction_T(input,model,clf_T):
    X_emb = model.encode(input, convert_to_numpy=True, normalize_embeddings=True)
    proba = clf_T.predict_proba(X_emb)[0]
    return proba



le = LabelEncoder()
y = le.fit_transform(['Hotel_info' ,'acceuil' ,'boiler_fixing' ,'chill_production',
 'chiller_fixing' ,'cogen_fixing' ,'fixing', 'get_consumption',
 'get_elec_consumption' ,'get_gaz_consumption' ,'get_occupation',
 'heat_production' ,'rooms' ,'temp_boiler' ,'temp_chill' ,'ventilation'])



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
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
# Accept user input

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
        proba = prediction_T(request, model,clf_T)
        pred_node,dic = main_node.next(proba, le, tree_json,[opt["next"] for opt in main_node.options])
        st.session_state.node_id = pred_node
        if main_node.type == "message":
            assistant_response = main_node.text
        if main_node.type == "action":
            assistant_response = main_node.on_action()
        if main_node.type == "form":
            assistant_response=main_node.texte
            if prompt := st.chat_input(main_node.text):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                # Display user message in chat message container
                with st.chat_message("user"):
                    st.markdown(prompt)
                assistant_response += str(main_node.on_action(prompt))
        if main_node.type == "handoff":
            assistant_response = main_node.text
        message_placeholder.markdown(assistant_response)
    # Add assistant response to chat history
    st.write(f"""classe reconnue : {pred_node} avec une proba de {max(dic["probabilités"])}""")
    dot = render_tree(tree_json, st.session_state["node_id"])
    st.graphviz_chart(dot, use_container_width=True)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
