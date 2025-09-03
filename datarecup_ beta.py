import time
import datetime
from datetime import datetime as dt
from datetime import timedelta

import json
import BAC0
import torch
import numpy as np
import os
import joblib
from sensorea_ia_lib import segment,search_sim,savefile,readandwrite,train




def predict(X,clf):
    input = X
    input = input.reshape(1, -1)
    y_pred = clf.predict(input)
    return y_pred

clf = joblib.load("./modele_multioutput_regression.pkl")

def readandwrite_model(model,path ="./data/trends_TH.json",pw=19,fw=1):
    with open(path, "r") as f:
        data_loaded = json.load(f)
    temp = data_loaded["temp_ext"][-(pw+fw):]
    data = np.array(data_loaded["commandes"])
    data = list(data.reshape(-1))

    # prediction creation du segment actuellement hybride pcq on a pas les conso mais il manque que ça

    conso_gaz = data_loaded["gaz consumption"][-(pw+fw):]
    conso_water = data_loaded["water consumption"][-(pw+fw):]


    t = data_loaded["time"]
    data.append(conso_gaz[pw])
    for i in temp:
        data.append(i)

    data.append(conso_water[-1])
    data.append(t)
    cons = [conso_gaz[i]-conso_gaz[i-1] for i in range(len(conso_gaz)-(pw+fw),len(conso_gaz),1)]
    for i in cons:
        data.append(i)
    X = np.array(data)
    y_pred = predict(X, clf)

    val_analogique = 0
    for i in range(len(y_pred[0])):
        print(i)
        val_analogique += y_pred[0][i] * (2 ** i)
    write(val_analogique)
    print(val_analogique)
    time_w = []
    delta = timedelta(days=0, hours=0, minutes=15, seconds=0)
    t = dt.now()
    for i in range(pw + fw, 0, -1):
        time_w.append(t - i * delta)
    cmd = np.reshape(X[0:240], (pw + fw, 12))
    temp_ext = list(X[241:261])
    w_cons = X[261]
    gaz_cons = X[263:]
    print(time_w)
    seg = segment(cmd, temp_ext, time_w, w_cons, gaz_cons, pw)
    path = savefile(seg,y_pred,pw)
    return path
pw = 19
fw = 1

root = "./dataset/"


def writing_trends(data):
    print("writting...✒️")
    for i in data:
        print(type(i))
    with open("./data/trends_TH.json", "w+") as f:
        json.dump(data, f, indent=4)
        f.close()

    print(f"{data} written to ./data/trends_TH.json")

with open("./data/IP.txt","r") as f:
    IPadress = str(f.read())
    print(IPadress)
    f.close()

bacnet = BAC0.lite(IPadress,bbmdAddress='192.168.1.101',bbmdTTL=900)
BAC0.log_level("silence")
w = 20
def write(val):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    ES_2807664['output_model_bin']=val
def get_room():
    dev = BAC0.device('192.168.1.50',1000, bacnet)
    objects = dev.points
    room_json = {}
    print(len(objects))
    for i in range(0,len(objects)-5,6):
        temp = str(objects[i]).split("-")
        temp = int(temp[0].replace("MCS BACnet stack\n/TH_CO2_", ""))
        room_json[str(temp)] = [str(objects[i+1]),str(objects[i+2]),str(objects[i+3]),str(objects[i+4]),str(objects[i+5])]
        print(room_json[str(temp)])

    return room_json
def get_cmds(cmds):
    chaufferie = BAC0.device('192.168.1.17',2886551,bacnet) #RPC_March 1.7.1 ; AS-27-2 BACnet Interface 2020 Saison 2
    objects = chaufferie.points
    for i in objects:
        print(i)
    temp = []
    temp.append(chaufferie['+2 return temp'])
    temp.append(chaufferie['+2 supply temp'])
    temp.append(chaufferie['+27 return temp'])
    temp.append(chaufferie['+27 supply temp'])
    temp.append(chaufferie['-4 return temp'])
    temp.append(chaufferie['-4 supply temp'])
    temp.append(chaufferie['boiler 1'])
    temp.append(chaufferie['boiler 2'])
    temp.append(chaufferie['boiler 3'])
    temp.append(chaufferie['cogen return temp'])
    temp.append(chaufferie['cogen supply temp'])

    temp.append(chaufferie['-4 pmpA mod'])
    temp.append(chaufferie['-4 pmpB mod'])
    temp.append(chaufferie['2 pmpA mod'])
    temp.append(chaufferie['2 pmpB mod'])
    temp.append(chaufferie['1 pmpA mod'])
    temp.append(chaufferie['1 pmpB mod'])
    temp.append(chaufferie['boiler1 mod'])
    temp.append(chaufferie['boiler2 mod'])
    temp.append(chaufferie['boiler3 mod'])
    cmd = []
    for i in range(0,len(temp)-9,1):
        t = str(temp[i])
        print(t)
        t = t.split(":")
        t = t[1]
        t = t.split(" ")
        t = float(t[1])
        cmd.append(t)
    tmp = []
    for i in range(len(temp)-9,len(temp),1):
        t = str(temp[i])
        print(t)
        t = t.split(":")
        t = t[1]
        t = t.split(" ")
        t = float(t[1])
        tmp.append(t)
    cmd.append((tmp[0]+tmp[1])/100)
    cmd.append((tmp[2]+tmp[3])/100)
    cmd.append((tmp[4]+tmp[5])/100)
    cmd.append((tmp[6]+tmp[7]+tmp[8])/300)
    cmds.append(cmd)
    return cmds
def get_gaz_conso(gaz):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        if "ES_2807664/gaz_conso" in str(i):
            t = str(i)
            print(t)
            t = t.split(" ")
            temp = float(t[2])
            gaz.append(temp)
            break
    return gaz
def get_water_conso(water):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        print(i)
        if "ES_2807664/water_conso" in str(i):
            t = str(i)
            print(t)
            t = t.split(" ")
            temp = float(t[2])
            water.append(temp)
            break
    return water
def get_elec_conso(elec):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        print(i)
        if "ES_2807664/elec_conso" in str(i):
            t = str(i)
            print(t)
            t = t.split(" ")
            temp = float(t[2])
            elec.append(temp)
            break
    return elec
def get_time():
    return datetime.datetime.now().hour
def get_occupation(occupation_list):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        if "ES_2807664/occupation" in str(i):
            t = str(i)
            print(t)
            t = t.split(" ")
            occ = float(t[2])
            occupation_list.append(occ)
            break
    return occupation_list
def get_true_occupation(True_occupation_list):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        if "ES_2807664/Effective occupation" in str(i):
            t = str(i)
            print(t)
            t = t.split(" ")
            occ = float(t[3])
            True_occupation_list.append(occ)
            break
    return True_occupation_list
def get_temp(temp_list):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    objects = ES_2807664.points
    for i in objects:
        if "ES_2807664/temp_ext" in str(i):
            t = str(i)
            t = t.split(" ")
            tmp = float(t[2])
            temp_list.append(tmp)
    return temp_list


def check_train():
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    return ES_2807664["train"]



with open("./data/trends_TH.json", "r") as f:
    data_loaded = json.load(f)

cmds_list = data_loaded['commandes']
water = data_loaded['water consumption']
gaz = data_loaded['gaz consumption']
elec = data_loaded['electricity consumption']
temp_list=data_loaded['temp_ext']
occupation_list = data_loaded['occupation_list']
rooms = data_loaded['rooms']
True_occupation_list = data_loaded['True_occupation_list']
t = 25
min = timedelta(days=0, hours=0, minutes=15, seconds=0)
t1 = datetime.datetime.today()
t1b = datetime.datetime.today().hour-1
data = {"occupation_list":occupation_list,
        "True_occupation_list":True_occupation_list,
        "water consumption":water,
        "electricity consumption":elec,
        "gaz consumption":gaz,
        "temp_ext":temp_list,
        'commandes': cmds_list,
        "time":t,
        "rooms":rooms}


while True:

    #padding d'initiation
    if len(cmds_list)<w:
        print("padding cmd")
        while (len(cmds_list)<w):
            cmds_list = get_cmds(cmds_list)

    if len(occupation_list)<(96):
        print("padding occupation")
        while (len(occupation_list)<(96)):
            occupation_list = get_occupation(occupation_list)
    if len(True_occupation_list)<(96):
        print("padding occupation")
        while (len(True_occupation_list)<(96)):
            True_occupation_list = get_true_occupation(True_occupation_list)

    if len(temp_list)<48:
        print("padding temp")
        while (len(temp_list)<48):
            temp_list = get_temp(temp_list)
    if len(gaz)<192:
        print("padding gaz")
        while (len(gaz)<192):
            gaz = get_gaz_conso(gaz)
    if len(elec)<192:
        print("padding elec")
        while (len(elec)<192):
            elec = get_elec_conso(elec)
    if len(water)<192:
        print("padding water")
        while (len(water)<192):
            water = get_water_conso(water)
    t2 = datetime.datetime.today()
    if (t2-t1)>=min:
        #ajout conso gaz
        gaz = gaz[1:]
        gaz = get_gaz_conso(gaz)
        data['gaz consumption'] = gaz
        #ajout conso elec
        elec = elec[1:]
        elec = get_elec_conso(elec)
        data['electricity consumption'] = elec
        # ajout conso water
        water = water[1:]
        water = get_water_conso(water)
        data['water consumption'] = water
        #ajout time
        T = get_time()
        data['time'] = T
        #ecriture des commandes
        t1 = datetime.datetime.today()
        cmds_list = cmds_list[1:]
        cmds_list = get_cmds(cmds_list)
        data['commandes'] = cmds_list
        #ecriture des trends
        occupation_list = occupation_list[1:]
        occupation_list = get_occupation(occupation_list)
        data['occupation_list'] = occupation_list
        True_occupation_list = True_occupation_list[1:]
        True_occupation_list = get_true_occupation(True_occupation_list)
        data['True_occupation_list'] = True_occupation_list
        data['rooms']=get_room()
        writing_trends(data)
        #rajout de l'échantillon dans la base de donnée
        path = readandwrite_model(clf)
        #ligne à commenter si on veut vraiment rajouter

    t2b = datetime.datetime.today().hour
    if t2b!=t1b:
        print("here i m")
        #ajout température
        temp_list = temp_list[1:]
        temp_list = get_temp(temp_list)
        data['temp_ext'] = temp_list
        t1b = datetime.datetime.today().hour
    if check_train()==True:
        train()