import time
import datetime
from datetime import datetime as dt
from datetime import timedelta

import json
import BAC0
import torch
import numpy as np
import os
from add_segment import segment,search_sim,savefile,readandwrite

pw = 19
fw = 1

root = "./dataset/"
dataset = []
for dirpath, dirnames, filenames in os.walk(root):
    for filename in filenames:
        if filename.endswith(".pt"):

                path = root + "/" + filename
                print(path)
                t = filename
                date = t[:8]
                date = date.replace("_", "-")
                heure = t[8:-3]
                heure = heure.replace("_", ":")
                t = dt.strptime(date+" "+heure, "%d-%m-%y %H:%M:%S")

                time_w = []
                delta = timedelta(days=0, hours=0, minutes=15, seconds=0)
                for i in range(pw+fw,0,-1):
                    time_w.append(t-i*delta)
                print(time_w[pw])
                data = torch.load(path,weights_only=False)
                input_array = data["input"]
                print(np.shape(input_array))
                input = input_array.reshape(-1)
                input = input.numpy()
                target = data["output"]
                target = target.numpy()
                cmd = np.reshape(input[0:240],(pw+fw,12))
                temp_ext = list(input[241:261])
                w_cons = input[261]
                time = input[262]
                gaz_cons = list(input[263:])
                seg = segment(cmd,temp_ext,time_w,w_cons,gaz_cons,pw)
                dataset.append(seg)
print(np.shape(dataset))

def writing_trends(data):
    print("writting...✒️")
    for i in data:
        print(type(i))
    with open("./data/trends.json", "w+") as f:
        json.dump(data, f, indent=4)
        f.close()

    print(f"{data} written to ./data/trends.json")


bacnet = BAC0.connect('172.21.212.141')
w = 20

def get_cmds(cmds):
    chaufferie = BAC0.device('192.168.1.17',2886551,bacnet) #RPC_March 1.7.1 ; AS-27-2 BACnet Interface 2020 Saison 2
    objects = chaufferie.points
    temp = []
    for i in objects:
        t = str(i)
        t = t.split(" ")
        cmd = t[2] in ["True", "true"]
        temp.append(int(cmd))
    cmds.append(temp)
    return cmds
def get_gaz_conso(gaz):
    test = torch.load('C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/temp.pt', weights_only=False)
    X = test['input']
    X = X.reshape(-1)
    X = list(X)
    conso = X[282]
    conso_list = conso.item()
    print(conso_list)
    gaz.append(conso_list)
    return gaz
def get_water_conso(water):
    test = torch.load('C:/Users/hugom/OneDrive/Documents/Stage_2025/dev_Cnn/dataset/temp.pt', weights_only=False)
    X = test['input']
    X = X.reshape(-1)
    X = list(X)
    water.append(X[261].item())
    return water
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
def degres_heure_glissants(temperatures, t_base=15.0):
    dh_glissant = sum(t_base - t for t in temperatures)
    return dh_glissant
def write(val):
    ES_2807664 = BAC0.device('192.168.1.101', 2807664, bacnet)
    ES_2807664['degré jour']=val


elec = []



dj_list = [-20.0, -13.333333333333334, -10.0, -8.0, -6.666666666666667, -5.714285714285714, -5.0, -4.444444444444445, -4.0, -42.0]
cmds_list = []
water = []
gaz = [40.62, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86]
temp_list=[17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0, 17.0]
occupation_list = [40.62, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86, 40.86]

t = 25
min = timedelta(days=0, hours=0, minutes=0, seconds=1)
t1 = datetime.datetime.today()
t1b = datetime.datetime.today().hour-1
data = {"occupation_list":occupation_list,
        "water consumption":water,
        "electricity consumption":elec,
        "gaz consumption":gaz,
        "temp_ext":temp_list,
        "dj":dj_list,
        'commandes': cmds_list,
        "time":t}

occupation_list = get_occupation([])
while True:
    #padding d'initiation
    if len(cmds_list)<w:
        while (len(cmds_list)<w):
            cmds_list = get_cmds(cmds_list)
    if len(occupation_list)<(2*w+1):
        while (len(occupation_list)<(2*w+1)):
            occupation_list = get_occupation(occupation_list)
    if len(temp_list)<w:
        while (len(temp_list)<w):
            temp_list = get_temp(temp_list)
    if len(gaz)<(2*w+1):
        while (len(gaz)<(2*w+1)):
            gaz = get_gaz_conso(gaz)
    if len(water)<(2*w+1):
        while (len(water)<(2*w+1)):
            water = get_water_conso(water)
    if len(dj_list)<(w/2):
        while (len(dj_list)<(w/2)):
            dj_list.append(degres_heure_glissants(temp_list)/(len(dj_list)+1))
    t2 = datetime.datetime.today()
    if (t2-t1)>=min:
        #ajout conso gaz
        gaz = gaz[1:]
        gaz = get_gaz_conso(gaz)
        data['gaz consumption'] = gaz
        # ajout conso water
        water = water[1:]
        water = get_water_conso(water)
        data['water consumption'] = water
        #ajout time
        T = get_time()
        data['time'] = T
        #ajout température
        temp_list = temp_list[1:]
        temp_list = get_temp(temp_list)
        data['temp_ext'] = temp_list
        #ecriture des commandes
        t1 = datetime.datetime.today()
        cmds_list = cmds_list[1:]
        cmds_list = get_cmds(cmds_list)
        data['commandes'] = cmds_list
        #ecriture des trends
        occupation_list = occupation_list[1:]
        occupation_list = get_occupation(occupation_list)
        data['occupation_list'] = occupation_list
        writing_trends(data)
        #rajout de l'échantillon dans la base de donnée
        path = readandwrite(dataset=dataset)
        #ligne à commenter si on veut vraiment rajouter 
        os.remove(path + ".pt")
    t2b = datetime.datetime.today().hour
    if t2b!=t1b:
        print("here i m")
        #ajout température
        temp_list = temp_list[1:]
        temp_list = get_temp(temp_list)
        t1b = datetime.datetime.today().hour
        dj = degres_heure_glissants(temp_list)
        dj_list = dj_list[1:]
        dj_list.append(dj)
        print(dj_list)
        data['dj'] = dj_list
        writing_trends(data)