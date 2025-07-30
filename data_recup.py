import time
import datetime
from datetime import timedelta
import json
import BAC0

def writing_trends(data):
    print("writting...✒️")
    with open("C:/Users\hugom\OneDrive\Documents\Stage_2025\data/trends.json", "w+") as f:
        json.dump(data, f, indent=4)
        f.close()
    print(f"{data} written to C:/Users\hugom\OneDrive\Documents\Stage_2025\data/trends.json")


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

water = []
elec = []
gaz = []
dj_list = []
cmds_list = []

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
        'commandes': cmds_list}

occupation_list = get_occupation([])
while True:
    #padding d'initiation
    if len(cmds_list)<w:
        while (len(cmds_list)<w):
            cmds_list = get_cmds(cmds_list)
    if len(occupation_list)<w:
        while (len(occupation_list)<w):
            occupation_list = get_occupation(occupation_list)
    if len(temp_list)<w:
        while (len(temp_list)<w):
            temp_list = get_temp(temp_list)
    if len(dj_list)<(w/2):
        while (len(dj_list)<(w/2)):
            dj_list.append(degres_heure_glissants(temp_list)/(len(dj_list)+1))
    t2 = datetime.datetime.today()
    if (t2-t1)>=min:
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



#%%
