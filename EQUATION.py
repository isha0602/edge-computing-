import numpy as np  
import pandas as pd 
from numpy import random 
import csv

df = pd.read_excel('F://dataset//iot/dataset.xlsx')
a=df['data_KB']
b = df['fog_MIPS']
c = df['cloud_MIPS']
d = df['input_size']
e = df['output_size']
g = df['up_link_BMPS']
h = df['down_link_BW']
q = df['app_id']
power =10
#TRANSMISSION TIME
txn=d/g
#RECIEVING TIME 
rxn = e/h
#exceution time for cloud server 
exe_t_c=a/c
#exceution delay for edge servers 
exe_t_f=a/b
#delay for cloud server
delay_c = txn+rxn+exe_t_c
# delay for edge server 
delay_e = txn+rxn+exe_t_f 
# computtaion cost for edge 
energy_e = power*delay_e

#computaion cost for cloud 
energy_c = power*delay_c

#Total cost
energy_a=txn+rxn+energy_e
energy_b= txn+rxn+energy+c 
total_cost=(energy_a+energy_b)/2
#Total delay 
total_delay =(delay_e+delay_c)/2
D = total_delay-txn-rxn 
print(D.mean())

with open('F://dataset//iot//data1.csv','w') as f:
  writer = csv.writer(f)
  header='TXN','RXN','EXE_T_F','EXE_T_C','DELAY_F','DELAY_C','ENERGY_C','ENERGY_F','ENERGY_A','ENERGY_B','APP_Id'
  writer.writerow(header)
  for i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11in zip(TXN,RXN,EXE_T_F,EXE_T_C,DELAY_F,DELAY_C,ENERGY_A,ENERGY_B,ENERGY_F,ENERGY_C,APPId):
    writer.writerow([i1]+[i2]+[i3]+[i4]+[i5]+[i6]+[i7]+[i8]+[i9]+[i10]+[i11]


