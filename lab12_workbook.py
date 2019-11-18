# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:09:04 2019

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

gasWellID = np.random.randint(1,17,5) 
#2,3,4,9,10

# Primary and Secondary Y-axes
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(df1['time'], df1['rate'], color="green", ls='None', marker='o', markersize=5,)
ax2.plot(df1['time'], df1['Cum']/1000, 'g-')
ax1.set_xlabel('Time, Months')
ax1.set_ylabel('Production Rate, bopm', color='g')
ax2.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax3 = plt.subplots()
ax4 = ax3.twinx()
ax3.plot(df2['time'], df2['rate'], color="green", ls='None', marker='o', markersize=5,)
ax4.plot(df2['time'], df2['Cum']/1000, 'g-')
ax3.set_xlabel('Time, Months')
ax3.set_ylabel('Production Rate, bopm', color='g')
ax4.set_ylabel('Cumulative Gas Production, Mbbls', color='b')
plt.show()

fig, ax5 = plt.subplots()
ax6 = ax5.twinx()
ax5.plot(df3['time'], df3['rate'], color="green", ls='None', marker='o', markersize=5,)
ax6.plot(df3['time'], df3['Cum']/1000, 'g-')
ax5.set_xlabel('Time, Months')
ax5.set_ylabel('Production Rate, bopm', color='g')
ax5.set_ylabel('Cumulative Gas Production, Mbbls', color='b')
plt.show()

fig, ax7 = plt.subplots()
ax8 = ax7.twinx()
ax7.plot(df4['time'], df4['rate'], color="green", ls='None', marker='o', markersize=5,)
ax8.plot(df4['time'], df4['Cum']/1000, 'g-')
ax7.set_xlabel('Time, Months')
ax7.set_ylabel('Production Rate, bopm', color='g')
ax8.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax9 = plt.subplots()
ax10 = ax9.twinx()
ax9.plot(df5['time'], df5['rate'], color="green", ls='None', marker='o', markersize=5,)
ax10.plot(df5['time'], df5['Cum']/1000, 'g-')
ax9.set_xlabel('Time, Months')
ax9.set_ylabel('Production Rate, bopm', color='g')
ax10.set_ylabel('Cumulative Gas Production, Mbbls', color='b')
plt.show()

fig, ax11 = plt.subplots()
ax12 = ax11.twinx()
ax11.plot(df5['time'], df5['rate'], color="green", ls='None', marker='o', markersize=5,)
ax12.plot(df5['time'], df5['Cum']/1000, 'g-')
ax11.set_xlabel('Time, Months')
ax11.set_ylabel('Production Rate, bopm', color='g')
ax12.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax13 = plt.subplots()
ax14 = ax13.twinx()
ax13.plot(df6['time'], df6['rate'], color="green", ls='None', marker='o', markersize=5,)
ax14.plot(df6['time'], df6['Cum']/1000, 'g-')
ax13.set_xlabel('Time, Months')
ax13.set_ylabel('Production Rate, bopm', color='g')
ax14.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax15 = plt.subplots()
ax16 = ax15.twinx()
ax15.plot(df7['time'], df7['rate'], color="green", ls='None', marker='o', markersize=5,)
ax16.plot(df7['time'], df7['Cum']/1000, 'g-')
ax15.set_xlabel('Time, Months')
ax15.set_ylabel('Production Rate, bopm', color='g')
ax16.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax17 = plt.subplots()
ax18 = ax17.twinx()
ax17.plot(df8['time'], df8['rate'], color="green", ls='None', marker='o', markersize=5,)
ax18.plot(df8['time'], df8['Cum']/1000, 'g-')
ax17.set_xlabel('Time, Months')
ax17.set_ylabel('Production Rate, bopm', color='g')
ax18.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax19 = plt.subplots()
ax20 = ax19.twinx()
ax19.plot(df9['time'], df9['rate'], color="green", ls='None', marker='o', markersize=5,)
ax20.plot(df9['time'], df9['Cum']/1000, 'g-')
ax19.set_xlabel('Time, Months')
ax19.set_ylabel('Production Rate, bopm', color='g')
ax20.set_ylabel('Cumulative Gas Production, Mbbls', color='b')
plt.show()

fig, ax21 = plt.subplots()
ax22 = ax21.twinx()
ax21.plot(df10['time'], df10['rate'], color="green", ls='None', marker='o', markersize=5,)
ax22.plot(df10['time'], df10['Cum']/1000, 'g-')
ax21.set_xlabel('Time, Months')
ax21.set_ylabel('Production Rate, bopm', color='g')
ax22.set_ylabel('Cumulative Gas Production, Mbbls', color='b')
plt.show()

fig, ax23 = plt.subplots()
ax24 = ax23.twinx()
ax23.plot(df11['time'], df11['rate'], color="green", ls='None', marker='o', markersize=5,)
ax24.plot(df11['time'], df11['Cum']/1000, 'g-')
ax23.set_xlabel('Time, Months')
ax23.set_ylabel('Production Rate, bopm', color='g')
ax24.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax25 = plt.subplots()
ax26 = ax25.twinx()
ax25.plot(df12['time'], df12['rate'], color="green", ls='None', marker='o', markersize=5,)
ax26.plot(df12['time'], df12['Cum']/1000, 'g-')
ax25.set_xlabel('Time, Months')
ax25.set_ylabel('Production Rate, bopm', color='g')
ax26.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax27 = plt.subplots()
ax28 = ax27.twinx()
ax27.plot(df13['time'], df13['rate'], color="green", ls='None', marker='o', markersize=5,)
ax28.plot(df13['time'], df13['Cum']/1000, 'g-')
ax27.set_xlabel('Time, Months')
ax27.set_ylabel('Production Rate, bopm', color='g')
ax28.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax29 = plt.subplots()
ax30 = ax29.twinx()
ax29.plot(df14['time'], df14['rate'], color="green", ls='None', marker='o', markersize=5,)
ax30.plot(df14['time'], df14['Cum']/1000, 'g-')
ax29.set_xlabel('Time, Months')
ax29.set_ylabel('Production Rate, bopm', color='g')
ax30.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax31 = plt.subplots()
ax32 = ax31.twinx()
ax31.plot(df15['time'], df15['rate'], color="green", ls='None', marker='o', markersize=5,)
ax32.plot(df15['time'], df15['Cum']/1000, 'g-')
ax31.set_xlabel('Time, Months')
ax31.set_ylabel('Production Rate, bopm', color='g')
ax32.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax33 = plt.subplots()
ax34 = ax33.twinx()
ax33.plot(df16['time'], df16['rate'], color="green", ls='None', marker='o', markersize=5,)
ax34.plot(df16['time'], df16['Cum']/1000, 'g-')
ax33.set_xlabel('Time, Months')
ax33.set_ylabel('Production Rate, bopm', color='g')
ax34.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

fig, ax35 = plt.subplots()
ax36 = ax35.twinx()
ax35.plot(df17['time'], df17['rate'], color="green", ls='None', marker='o', markersize=5,)
ax36.plot(df17['time'], df17['Cum']/1000, 'g-')
ax35.set_xlabel('Time, Months')
ax35.set_ylabel('Production Rate, bopm', color='g')
ax36.set_ylabel('Cumulative Oil Production, Mbbls', color='b')
plt.show()

#stacked plots
wellID = 1
df1 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 2
df2 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 3
df3 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 
wellID = 4
df4 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 5
df5 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 6
df6 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 
wellID = 7
df7 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 8
df8 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 9
df9 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 
wellID = 10
df10 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 11
df11 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 12
df12 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 13
df13 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 14
df14 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 15
df15 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn) 
wellID = 16
df16 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
wellID = 17
df17 = pd.read_sql_query(f"SELECT time,rate, Cum,Cum_model FROM Rates WHERE wellID={wellID};", conn)
    
    
labels = ["well 2", "well 3", "well 4","well 9", "well 10"]
fig, ax = plt.subplots()
ax.stackplot(df1['time'], df2['Cum']/1000, df3['Cum']/1000,df4['Cum']/1000, df9['Cum']/1000, df10['Cum']/1000, labels=labels)
ax.legend(loc='upper left')
plt.title("Historical Field Gas Production Rate")
plt.show()

labels = ["well 1", "well 5", "well 6", "well 7", "well 8", "well 11", "well 12", "well 13", "well 14", "well 15", "well 16", "well 17"]
fig, ax = plt.subplots()
ax.stackplot(df1['time'], df1['Cum']/1000, df5['Cum']/1000,df6['Cum']/1000, df7['Cum']/1000, df8['Cum']/1000,df11['Cum']/1000, df12['Cum']/1000,df13['Cum']/1000, df14['Cum']/1000, df15['Cum']/1000,df16['Cum']/1000, df17['Cum']/1000, labels=labels)
ax.legend(loc='upper left')
plt.title("Historical Field Oil Production Rate")
plt.show()

#stacked bar graph
N = 6
ind = np.arange(1,N+1)    # the x locations for the groups
months = ['Jan','Feb','Mar','Apr','May','Jun']
width = 0.5

p1 = plt.bar(df1['time'][0:N], df1['Cum'][0:N]/1000, width)
p2 = plt.bar(df1['time'][0:N], df5['Cum'][0:N]/1000, width, bottom=df1['Cum'][0:N]/1000)
p3 = plt.bar(df1['time'][0:N], df6['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N])/1000)
p4 = plt.bar(df1['time'][0:N], df7['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N])/1000)
p5 = plt.bar(df1['time'][0:N], df8['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N])/1000)
p6 = plt.bar(df1['time'][0:N], df11['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N])/1000)
p7 = plt.bar(df1['time'][0:N], df12['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N])/1000)
p8 = plt.bar(df1['time'][0:N], df13['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N])/1000)
p9 = plt.bar(df1['time'][0:N], df14['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N])/1000)
p10 = plt.bar(df1['time'][0:N], df15['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N]+df14['Cum'][0:N])/1000)
p11 = plt.bar(df1['time'][0:N], df16['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N]+df14['Cum'][0:N]+df15['Cum'][0:N])/1000)
p12 = plt.bar(df1['time'][0:N], df17['Cum'][0:N]/1000, width, bottom=(df1['Cum'][0:N]+df5['Cum'][0:N]+ df6['Cum'][0:N] +df7['Cum'][0:N] +df8['Cum'][0:N]+df11['Cum'][0:N]+df12['Cum'][0:N]+df13['Cum'][0:N]+df14['Cum'][0:N]+df15['Cum'][0:N]+df16['Cum'][0:N])/1000)           

                            
plt.ylabel('Oil Production, Mbbls')
plt.title('Cumulative Oil Production Forecast')
plt.xticks(ind, months, fontweight='bold')
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0]), ("well 1", "well 5", "well 6", "well 7", "well 8", "well 11", "well 12", "well 13", "well 14", "well 15", "well 16", "well 17"))
plt.show()

p13 = plt.bar(df2['time'][0:N], df2['Cum'][0:N]/1000, width)
p14 = plt.bar(df2['time'][0:N], df3['Cum'][0:N]/1000, width, bottom=df2['Cum'][0:N]/1000)
p15 = plt.bar(df2['time'][0:N], df4['Cum'][0:N]/1000, width, bottom=(df2['Cum'][0:N]+df3['Cum'][0:N])/1000)
p16 = plt.bar(df2['time'][0:N], df9['Cum'][0:N]/1000, width, bottom=(df2['Cum'][0:N]+df3['Cum'][0:N]+ df4['Cum'][0:N])/1000)
p17 = plt.bar(df2['time'][0:N], df10['Cum'][0:N]/1000, width, bottom=(df2['Cum'][0:N]+df3['Cum'][0:N]+ df4['Cum'][0:N] +df9['Cum'][0:N])/1000)

plt.ylabel('Gas Production, Mbbls')
plt.title('Cumulative Gas Production Forecast')
plt.xticks(ind, months, fontweight='bold')
plt.legend((p13[0], p14[0], p15[0], p16[0], p17[0]), ("well 2", "well 3", "well 4", "well 9", "well 10"))
plt.show()



#well log tracks
data1 = np.loadtxt("volve_logs/15_9-F-4_INPUT.LAS", skiprows=69)
data1DF = pd.read_csv("volve_logs/15_9-F-4_INPUT.LAS",skiprows=69, sep = '\s+' )

#data2 = np.loadtxt("volve_logs/volve_logs/15_9-F-1B_INPUT.LAS", skiprows=69)
#data2DF = pd.read_csv("volve_logs/volve_logs/15_9-F-1B_INPUT.LAS",skiprows=69, sep = '\s+' )

#data3 = np.loadtxt("volve_logs/volve_logs/15_9-F-14_INPUT.LAS", skiprows=69)
#data3DF = pd.read_csv("volve_logs/volve_logs/15_9-F-14_INPUT.LAS",skiprows=69, sep = '\s+' )

data = np.loadtxt("WLC_PETRO_COMPUTED_INPUT_1.DLIS.0.las", skiprows=48)
DZ,rho=data[:,0], data[:,1]
DZ=DZ[np.where(rho>0)]
rho=rho[np.where(rho>0)]

print('Investigated Depth',[min(DZ),max(DZ)])

titleFontSize = 22
fontSize = 20
#Plotting multiple well log tracks on one graph
fig = plt.figure(figsize=(36,20),dpi=100)
fig.tight_layout(pad=1, w_pad=4, h_pad=2)

plt.subplot(1, 6, 1)
plt.grid(axis='both')
plt.plot(rho,DZ, color='red')
plt.title('DT vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DT, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 2)
plt.grid(axis='both')
plt.plot(rho,DZ, color='green')
plt.title('DTS vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('DTS, us/ft', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 3)
plt.grid(axis='both')
plt.plot(rho,DZ, color='blue')
plt.title('GR vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('GR, API', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 4)
plt.grid(axis='both')
plt.plot(rho,DZ, color='black')
plt.title('NPHI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('NPHI, v/v', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

plt.subplot(1, 6, 5)
plt.grid(axis='both')
plt.plot(rho,DZ, color='brown')
plt.title('CALI vs Depth', fontsize=titleFontSize, fontweight='bold')
plt.xlabel('CALI, inches', fontsize = fontSize, fontweight='bold')
plt.ylabel('Depth, m', fontsize = fontSize, fontweight='bold')
plt.gca().invert_yaxis()

fig.savefig('well_1_log.png', dpi=600)


