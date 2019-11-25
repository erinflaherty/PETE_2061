# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 15:38:43 2019

@author: erinf
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3

conn = sqlite3.connect("DCA.db") 
cur = conn.cursor()

titleFontSize = 18
axisLabelFontSize = 15
axisNumFontSize = 13


# Primary and Secondary Y-axes
for wellID in range(1,18):
    productionDF = pd.read_sql_query(f'SELECT time, rate, Cum, Cum_model FROM Rates WHERE wellID = {wellID};', conn)
    dcaDF = pd.read_sql_query(f'SELECT *FROM DCAparams;', conn)
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(productionDF['time'], productionDF['rate'], color="blue", ls='None', marker='o', markersize=5,)
    ax2.plot(productionDF['time'], productionDF['Cum']/1000, 'g-')
    ax1.set_xlabel('Time, Months')
    ax1.set_ylabel('Production Rate, bopm', color='g')
    ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
    plt.show()
    
#stacked plot
for wellID in range(1,18):
    productionDF = pd.read_sql_query(f"SELECT time FROM Rates WHERE wellID = {wellID};", conn)
    dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='gas';", conn)
j= 1
for i in dcaDF['wellID']:
    productionDF['Well'+ str(i)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID = {i};", conn)
    
production = productionDF.iloc[:,1:].values
time = productionDF['time'].values
labels = productionDF.columns
labels = list(labels[1:])
print(labels)
fig, ax = plt.subplots()
ax.stackplot(time,np.transpose(production), labels = labels)
ax.legend(loc = "upper right")
plt.title('Cumulative Gas Production')
plt.show()

for wellID in range(1,18):
    productionDF = pd.read_sql_query(f"SELECT time FROM Rates WHERE wellID = {wellID};", conn)
    dcaDF = pd.read_sql_query("SELECT wellID FROM DCAparams WHERE fluid='oil';", conn)
j= 1
for i in dcaDF['wellID']:
    productionDF['Well'+ str(i)] = pd.read_sql_query(f"SELECT rate FROM Rates WHERE wellID = {i};", conn)
    
production = productionDF.iloc[:,1:].values
time = productionDF['time'].values
labels = productionDF.columns
labels = list(labels[1:])
print(labels)
fig, ax = plt.subplots()
ax.stackplot(time,np.transpose(production), labels = labels)
ax.legend(loc = "upper right")
plt.title('Cumulative Oil Production')
plt.show()

#stacked bar graphs
N = 6
ind = np.arange(1,N+1)    
months = ['Jan','Feb','Mar','Apr','May','Jun']
width = 0.5
result = np.zeros(len(months))
labels = []
loc_plots = []
cumulativeDF = pd.DataFrame(productionDF["time"])
dcaDF = pd.read_sql_query(f"SELECT wellID FROM DCAparams WHERE fluid = 'gas';", conn)

for x in dcaDF['wellID']:
    cumulativeDF["well" + str(x)] = pd.read_sql_query(f" SELECT Cum FROM Rates WHERE wellID = {x};", conn)

j = 1
for i in dcaDF['wellID']:
    p1 = plt.bar(cumulativeDF['time'][0:N], cumulativeDF['well'+ str(x)][0:N]/1000, width, bottom = result)
    labels.append('well' + str(i))
    loc_plots.append(p1)
    plt.ylabel('Gas Production, Mbbls')
    plt.title('Cumulative Production Forecast')
    plt.xticks(ind, months, fontweight='bold')
    
    j+=1
    split = cumulativeDF.iloc[0:6, 1:j].values
    result = np.sum(a = split, axis = 1)/1000
plt.legend(loc_plots, labels)
plt.show(loc_plots)

N = 6
ind = np.arange(1,N+1)    
months = ['Jan','Feb','Mar','Apr','May','Jun']
width = 0.5
result = np.zeros(len(months))
labels = []
loc_plots = []
cumulativeDF = pd.DataFrame(productionDF["time"])
dcaDF = pd.read_sql_query(f"SELECT wellID FROM DCAparams WHERE fluid = 'oil';", conn)

for y in dcaDF['wellID']:
    cumulativeDF["well" + str(y)] = pd.read_sql_query(f" SELECT Cum FROM Rates WHERE wellID = {x};", conn)

k = 1
for n in dcaDF['wellID']:
    p1 = plt.bar(cumulativeDF['time'][0:N], cumulativeDF['well'+ str(y)][0:N]/1000, width, bottom = result)
    labels.append('well' + str(n))
    loc_plots.append(p1)
    plt.ylabel('Oil Production, Mbbls')
    plt.title('Cumulative Production Forecast')
    plt.xticks(ind, months, fontweight='bold')
    
    k+=1
    split = cumulativeDF.iloc[0:6, 1:k].values
    result = np.sum(a = split, axis = 1)/1000
plt.legend(loc_plots, labels)
plt.show(loc_plots)

#logs
d1 = np.loadtxt("C:/Users/erinf/Downloads/pete2061-master (5)/pete2061-master/volve_logs/volve_logs/15_9-F-1B_INPUT.LAS", skiprows = 69)
DZ1, rho1 = d1[:,0], d1[:,16]
DZ1 = DZ1[np.where(rho1>0)]
rho1 = rho1[np.where(rho1>0)]

titleFontSize=22
fontSize=20

fig=plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)
plt.subplot(1,6,1)
plt.grid(axis='both')
plt.plot(rho1,DZ1, color='green')
plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DT1 =d1[:, 0], d1[:,8]
DZ1=DZ1[np.where(DT1>0)]
DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)
plt.grid(axis='both')
plt.plot(DT1,DZ1, color='yellow')
plt.title('DT v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DTS1 =d1[:, 0], d1[:,9]
DZ1=DZ1[np.where(DTS1>0)]
DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)
plt.grid(axis='both')
plt.plot(DTS1,DZ1, color='blue')
plt.title('DTS v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, GR1 =d1[:, 0], d1[:,10]
DZ1=DZ1[np.where(GR1>0)]
GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)
plt.grid(axis='both')
plt.plot(GR1,DZ1, color='red')
plt.title('GR v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, NPHI1 =d1[:, 0], d1[:,12]
DZ1=DZ1[np.where(NPHI1>0)]
NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)
plt.grid(axis='both')
plt.plot(NPHI1,DZ1, color='purple')
plt.title('NPHI v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, CALI1 =d1[:, 0], d1[:,6]
DZ1=DZ1[np.where(CALI1>0)]
CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1,6,6)
plt.grid(axis='both')
plt.plot(CALI1,DZ1, color='black')
plt.title('Caliper v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()



d2=np.loadtxt("C:/Users/erinf/Downloads/pete2061-master (5)/pete2061-master/volve_logs/volve_logs/15_9-F-4_INPUT.las", skiprows=65)
DZ1, rho1= d2[:,0], d2[:,7]
DZ1= DZ1[np.where(rho1>0)]
rho1= rho1[np.where(rho1>0)]

fig=plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1,6,1)
plt.grid(axis='both')
plt.plot(rho1,DZ1, color='green')
plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DT1 =d2[:,0], d2[:,2]
DZ1=DZ1[np.where(DT1>0)]
DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)
plt.grid(axis='both')
plt.plot(DT1,DZ1, color='yellow')
plt.title('DT v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, ft', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DTS1 =d2[:, 0], d2[:,3]
DZ1=DZ1[np.where(DTS1>0)]
DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)
plt.grid(axis='both')
plt.plot(DTS1,DZ1, color='blue')
plt.title('DTS v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, GR1 =d2[:, 0], d2[:,4]
DZ1=DZ1[np.where(GR1>0)]
GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)
plt.grid(axis='both')
plt.plot(GR1,DZ1, color='red')
plt.title('GR v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, NPHI1 =d2[:, 0], d2[:,5]
DZ1=DZ1[np.where(NPHI1>0)]
NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)
plt.grid(axis='both')
plt.plot(NPHI1,DZ1, color='purple')
plt.title('NPHI v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, CALI1 =d2[:, 0], d2[:,6]
DZ1=DZ1[np.where(CALI1>0)]
CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1,6,6)
plt.grid(axis='both')
plt.plot(CALI1,DZ1, color='black')
plt.title('Caliper v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()


d3 = np.loadtxt("C:/Users/erinf/Downloads/pete2061-master (5)/pete2061-master/volve_logs/volve_logs/15_9-F-14_INPUT.las", skiprows = 69)
DZ1, rho1 = d3[:,0], d3[:,9]
DZ1 = DZ1[np.where(rho1>0)]
rho1 = rho1[np.where(rho1>0)]

titleFontSize=22
fontSize=20

fig=plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1,6,1)
plt.grid(axis='both')
plt.plot(rho1,DZ1, color='green')
plt.title('Density v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Density, g/cc', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DT1 =d3[:, 0], d3[:,3]
DZ1=DZ1[np.where(DT1>0)]
DT1=DT1[np.where(DT1>0)]

plt.subplot(1,6,2)
plt.grid(axis='both')
plt.plot(DT1,DZ1, color='yellow')
plt.title('DT v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, DTS1 =d3[:, 0], d3[:,4]
DZ1=DZ1[np.where(DTS1>0)]
DTS1=DTS1[np.where(DTS1>0)]

plt.subplot(1,6,3)
plt.grid(axis='both')
plt.plot(DTS1,DZ1, color='blue')
plt.title('DTS v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, GR1 =d3[:, 0], d3[:,5]
DZ1=DZ1[np.where(GR1>0)]
GR1=GR1[np.where(GR1>0)]

plt.subplot(1,6,4)
plt.grid(axis='both')
plt.plot(GR1,DZ1, color='red')
plt.title('GR v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, NPHI1 =d3[:, 0], d3[:,6]
DZ1=DZ1[np.where(NPHI1>0)]
NPHI1=NPHI1[np.where(NPHI1>0)]

plt.subplot(1,6,5)
plt.grid(axis='both')
plt.plot(NPHI1,DZ1, color='purple')
plt.title('NPHI v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

DZ1, CALI1 =d3[:, 0], d3[:,6]
DZ1=DZ1[np.where(CALI1>0)]
CALI1=CALI1[np.where(CALI1>0)]

plt.subplot(1,6,6)
plt.grid(axis='both')
plt.plot(CALI1,DZ1, color='black')
plt.title('Caliper v Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('Caliper, inch', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m ', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()














