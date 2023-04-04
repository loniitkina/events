import csv
import re
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from glob import glob
from event_func import getColumn

inpath='../../transect/data/SnowModel/final/'
outpath='../event_plots/'

#set up the time series plot
#GENERAL FIGURE - ax
#all products: total deformation, buoy area, LKF stats
#temperature
#wind

fig1, ax = plt.subplots(6, 1,gridspec_kw={'height_ratios': [1,1,1,.5,.3,1]},figsize=(10,18))
fig_start = datetime(2019,10,15)
fig_end = datetime(2020,5,15)

ax[0].set_ylabel('Total deformation (day$^{-1}$)', fontsize=15)
ax[0].tick_params(axis="x", labelsize=12)
ax[0].tick_params(axis="y", labelsize=12)
ax[0].set_xlim(fig_start,fig_end)

ax0 = ax[0].twinx()

ax[1].set_ylabel('Area (km$^2$)', fontsize=15)
ax[1].tick_params(axis="x", labelsize=12)
ax[1].tick_params(axis="y", labelsize=12)
ax[1].set_xlim(fig_start,fig_end)

ax1 = ax[1].twinx()
ax1.set_ylabel('Area change (%)', fontsize=15)

#ice stress and seismics
ax[2].set_ylabel('Relative ice stress', fontsize=15)
ax[2].tick_params(axis="x", labelsize=12)
ax[2].tick_params(axis="y", labelsize=12)
ax[2].set_xlim(fig_start,fig_end)

ax2 = ax[2].twinx()
ax2.set_ylabel('Ice seismics', fontsize=15, c='r')

#AFS statistics
ax[3].set_ylabel('LKF Area', fontsize=15)
ax[3].tick_params(axis="x", labelsize=12)
ax[3].tick_params(axis="y", labelsize=12)
ax[3].set_xlim(fig_start,fig_end)

ax3 = ax[3].twinx()
ax3.set_ylabel('DE count', fontsize=15, c='r')

#LKF angles


#air temperature
ax[4].set_ylabel('T$_{air}$ ($^\circ$C)', fontsize=15)
ax[4].tick_params(axis="x", labelsize=12)
ax[4].tick_params(axis="y", labelsize=12)
ax[4].set_xlim(fig_start,fig_end)

#wind speed and direction
ax[5].set_ylabel('Wind Speed (m/s)', fontsize=15)
ax[5].tick_params(axis="x", labelsize=12)
ax[5].tick_params(axis="y", labelsize=12)
ax[5].set_xlim(fig_start,fig_end)


#met data
fname = inpath+'final_10m_3hrly_met_2023_02_14.dat'
print(fname)

#dates
numdays=366*8
start = datetime(2019,8,1)
dt = [start + timedelta(hours=x*3) for x in range(numdays)]
end = datetime(2020,8,1)

results = csv.reader(open(fname))
#get rid of all multi-white spaces and split in those that remain
results_clean = [re.sub(" +", " ",row[0]) for row in results]
#temperature, humidity, wind speed, wind direction, precipitation
#model=bias-corrected MERRA-2 reanalysis

tair = [row.split(" ")[2] for row in results_clean]
tair = np.array(tair,dtype=np.float)     
tair = np.ma.array(tair,mask=tair==-9999)

tair_model = [row.split(" ")[3] for row in results_clean]
tair_model = np.array(tair_model,dtype=np.float)     
tair_model = np.ma.array(tair_model,mask=tair_model==-9999)

ax[4].plot(dt,tair_model,c='k')
ax[4].plot(dt,tair,c='darkred')
zeros=np.zeros_like(tair_model)
ax[4].plot(dt,zeros,c='k')

ws = [row.split(" ")[6] for row in results_clean]
ws = np.array(ws,dtype=np.float)     
ws = np.ma.array(ws,mask=ws==-9999)

ws_model = [row.split(" ")[7] for row in results_clean]
ws_model = np.array(ws_model,dtype=np.float)     
ws_model = np.ma.array(ws_model,mask=ws==-9999)

wd = [row.split(" ")[8] for row in results_clean]
wd = np.array(wd,dtype=np.float)     
wd = np.ma.array(wd,mask=wd==-9999)

wd_model = [row.split(" ")[9] for row in results_clean]
wd_model = np.array(wd_model,dtype=np.float)     
wd_model = np.ma.array(wd_model,mask=wd==-9999)

cs = ax[5].scatter(dt,ws_model,marker='.',c='k',s=2,zorder=100)
cs = ax[5].scatter(dt,ws,c=wd,cmap=plt.cm.twilight,s=5,zorder=200)
cb = plt.colorbar(cs,orientation='horizontal',aspect=80, fraction=.05, pad=.2, ax=ax[5])  # draw colorbar
cb.set_label(label='Wind direction (deg.)',fontsize=15)

#stress and seismics
inpath = '../data/FMI_stress/'
fnames = sorted(glob(inpath+'*.txt'))

#date,RECORD,CPU_Temp,Chan1_Freq,Chan1_Therm,Chan2_Freq,Chan3_Freq,p,q,theta
#In my analysis, I have used the the first principal stress 'p' which is the 8th column of the data file. 'q' is the second principal stress and theta angle between those stresses. 

for fname in fnames:
    print(fname)
    label = fname.split('_')[-1].split('.')[0]
    if label=='3691':
        label='Nloop'
    elif label=='3692':
        label='MET'
    elif label=='3693':
        label='Heli'
    elif label=='3694':
        label='Lead'
    else:
        label='new sensor'
    
    time = np.array(getColumn(fname,0))
    stress_dates = [ datetime.strptime(time[x], "%Y-%m-%d %H:%M:%S") for x in range(len(time)) ]

    pstress = getColumn(fname,7); pstress = np.array(pstress,dtype=np.float)
    qstress = getColumn(fname,8); qstress = np.array(qstress,dtype=np.float)
    theta = getColumn(fname,9); theta = np.array(theta,dtype=np.float)
    
    #theta is always between -90 and 90
    print(np.max(theta))
    print(np.min(theta))
    
    
    #total stress could be computed like this is the components were rectangular
    stress = np.sqrt(pstress**2+qstress**2)
    #but there is angle theta between them!
    
    pstress = stress
    
    #make realtive stress (these are non-calibrated values anyway)
    sm=np.mean(pstress)
    pstress=pstress/sm
    
    ax[2].plot(stress_dates,pstress,label=label)
    
    ax0.plot(stress_dates,pstress/100,c='0.8',alpha=0.5)
    
ax[2].set_ylim(0,5)
ax[2].legend(ncol=4)

#Findings: 
#1. There is thermal signal, but maybe only obvious end of Feb, beginning of March, when some silence, otherwise the deformation prevails.
#2. Stresses are very local and pairs can be extremly negativelly correlated in event-to-event bases
#3. Total relative stress shows good correspondence with the strain rates. Change in stress before and during the strain rate events.
#4. Stresses increase also during elastic and very small strain rates, that are not detectable by the GPS- and SAR-precision strain rates.

#Sensors are not even relativelly similar to each other - some sensors seem to be more sensitive than the others

#AFS statistics
inpath = '../data/AFS/'
fnames = glob(inpath+'afs_ship2019-2020_120km.csv')

afs_dates=[]
dec=[]
lkf=[]

for fname in fnames:
    print(fname)
    time = np.array(getColumn(fname,0))
    tmp = [ datetime.strptime(time[x], "%Y-%m-%d %H:%M:%S") for x in range(len(time)) ]

    dd = getColumn(fname,1); dd = np.array(dd,dtype=np.float)
    ll = getColumn(fname,8); ll = np.array(ll,dtype=np.float)
    
    afs_dates.extend(tmp)
    dec.extend(dd)
    lkf.extend(ll)
    
ax[3].plot(afs_dates,lkf,c='k',linestyle='None',marker='o',ms=3)
ax3.plot(afs_dates,dec,c='r',linestyle='None',marker='o',ms=3)

#sea ice deformation from buoys - Jenny's calculations
inpath = '../data/Buoys_JennyHutchings/'
fnames = sorted(glob(inpath+'strainrate_MOSAiC_DN_*.csv'))

for fname in fnames:
    print(fname)
    time = np.array(getColumn(fname,0))
    dt_jenny = [ datetime.strptime(time[x], "%Y-%m-%dT%H:%M:%S.%f") for x in range(len(time)) ]
    aa = getColumn(fname,9); aa = np.array(aa,dtype=np.float) 
    aa = aa/aa[0]*100   #relative area change in %
    
    #get total deformation
    dd = getColumn(fname,4); dd = np.array(dd,dtype=np.float)*24*3600
    ss = getColumn(fname,5); ss = np.array(ss,dtype=np.float)*24*3600
    tt = np.sqrt(dd**2+ss**2)
    
    idx=fname.split('_')[-3]
    if idx=='1':
        label='CO area'
        color='royalblue'
        th = .1
    elif idx=='2':
        label='L sites'
        color='teal'
        th=.1
    elif idx=='4':
        label='200-km region'
        color='gold'
        th=.05
    else:
        label=''
        color='.75'
        th=.1
    
    ax[0].plot(dt_jenny,tt,c=color,label=label,alpha=.7)
    ax1.plot(dt_jenny,aa,c=color,label=label,alpha=.7)

ax[0].set_ylim(0,1)
    
#sea ice deformation from SAR (Luisa)
inpath = '../data/SAR_Luisa_vonAlbedyll/'
fnames = sorted(glob(inpath+'Polarstern_2019-10-05_2020-07-14_*.txt'))

for fname in fnames:
    print(fname)
    time = np.array(getColumn(fname,1))
    dt_luisa = [ datetime.strptime(time[x], "%Y-%m-%d %H:%M:%S") for x in range(len(time)) ]
    
    #get total deformation
    dd = getColumn(fname,2); dd = np.array(dd,dtype=np.float)*24*3600
    ss = getColumn(fname,3); ss = np.array(ss,dtype=np.float)*24*3600
    tt = np.sqrt(dd**2+ss**2)
    
    tt_cum = np.cumsum(np.ma.fix_invalid(tt,fill_value=0))
        
    idx=fname.split('_')[-3]
    if idx=='100km':
        label=idx+' SAR'
        color='r'
        th = .1
    elif idx=='50km':
        label=idx+' SAR'
        color='darkred'
        th=.1
    else:
        label=''
        color='.75'
        th=.1
    
    ax0.plot(dt_luisa,tt,c=color,label=label,alpha=.7)
    

#dates for the publisher
from matplotlib.dates import MonthLocator, DateFormatter
import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

ax[0].xaxis.set_minor_locator(MonthLocator())
ax[0].xaxis.set_major_formatter(DateFormatter('%b %Y'))

ax[1].xaxis.set_minor_locator(MonthLocator())
ax[1].xaxis.set_major_formatter(DateFormatter('%b %Y'))

ax[2].xaxis.set_minor_locator(MonthLocator())
ax[2].xaxis.set_major_formatter(DateFormatter('%b %Y'))

ax[0].legend(loc='upper left',ncol=5)

plt.show()
outfig = outpath+'main_ts.png'
print(outfig)
fig1.savefig(outfig,bbox_inches='tight')
