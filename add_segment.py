import torch
import numpy as np
import os
from datetime import datetime
from datetime import timedelta
import json


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