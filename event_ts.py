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

#met data
fname = inpath+'final_10m_3hrly_met_2023_02_14.dat'
print(fname)

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

#dates
numdays=366*8
start = datetime(2019,8,1)
dt = [start + timedelta(hours=x*3) for x in range(numdays)]
end = datetime(2020,8,1)


##sea ice deformation from buoys - Polona's calculations
#inpath = '../../transect/data/mosaic_buoy_data/selection/'
#fname = inpath+'Deformation_3hr.csv'

#year = np.array(getColumn(fname,0),dtype=int)
#month = np.array(getColumn(fname,1),dtype=int)
#day = np.array(getColumn(fname,2),dtype=int)
##td_dates = [ datetime(year[x],month[x],day[x]) for x in range(0,len(year)) ]
##include 3-hour time lag for deformation (buoys show deformation for past hour)
#td_dates = [ datetime(year[x],month[x],day[x])+ timedelta(hours=3) for x in range(0,len(year)) ]

#td = getColumn(fname,7); td = np.array(td,dtype=np.float)
#td_cum = np.cumsum(td)

##define periods of several connected events
##use daily resolution (or 3-daily, or weekly)
##whenever dE/dt > threshold = event
#pos = {'td_cum': td_cum}
#df = pd.DataFrame(data=pos,index=td_dates)
#days = df.resample('D').sum().asfreq('D')
#td_cumD = days.td_cum.values
#days = days.td_cum.index
##print(days)
##convert back to datetime list
#days = days.astype('O')
#dtDp = td_cumD[1:]-td_cumD[:-1]
#periods = dtDp>1.5
#periods = np.where(dtDp>1.5,0,1)

#dt_periods = periods[1:]-periods[:-1]
##print(dt_periods)
#period_start = np.ma.array(days[2:],mask=dt_periods>-1).compressed()
#period_end = np.ma.array(days[2:],mask=dt_periods<1).compressed()
##print(period_start)
##print(period_end)
##exit()

##events (high temporal resolution)
#dtD = td_cum[1:]-td_cum[:-1]
#events = dtD>.1
#events = np.where(dtD>.1,0,1)

#dt_events = events[1:]-events[:-1]

#event_start = np.ma.array(td_dates[2:],mask=dt_events>-1).compressed()
#event_end = np.ma.array(td_dates[2:],mask=dt_events<1).compressed()

##print(event_start[:5])
##print(event_end[:5])
##exit()

##area of the buoy triangle(s)
#area = getColumn(fname,4); area = np.array(area,dtype=np.float)**2/1e6  #convert from m to km^2


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






#EVENTS FIGURE - bx
#cumulative total deformation
#ice stress
#LKF stats
#Wind
#distance to ice edge/land 
#Events shading

#Findings: there is no quincent period, the events do not observe sinoptic time scale, there are several short events inside the sinoptic events and then there are also events not connected to strong winds or wind direction change (or there is latency in sea ice drft direction change!)

fig2, bx = plt.subplots(5, 1,gridspec_kw={'height_ratios': [1,.5,.5,.5,.5]},figsize=(10,15))
fig_start = datetime(2019,10,15)
fig_end = datetime(2020,5,15)

bx[0].set_ylabel('$\epsilon_{tot}$', fontsize=15)
bx[0].tick_params(axis="x", labelsize=12)
bx[0].tick_params(axis="y", labelsize=12)
bx[0].set_xlim(fig_start,fig_end)

bx1 = bx[0].twinx()

bx[1].set_ylabel('ice stress', fontsize=15)
bx[1].tick_params(axis="x", labelsize=12)
bx[1].tick_params(axis="y", labelsize=12)
bx[1].set_xlim(fig_start,fig_end)

bx[2].set_ylabel('ice seismics', fontsize=15)
bx[2].tick_params(axis="x", labelsize=12)
bx[2].tick_params(axis="y", labelsize=12)
bx[2].set_xlim(fig_start,fig_end)

bx[4].set_ylabel('Wind Speed (m/s)', fontsize=15)
bx[4].tick_params(axis="x", labelsize=12)
bx[4].tick_params(axis="y", labelsize=12)
bx[4].set_xlim(fig_start,fig_end)

cs = bx[4].scatter(dt,ws_model,marker='.',c='k',s=2,zorder=100)
cs = bx[4].scatter(dt,ws,c=wd,cmap=plt.cm.twilight,s=5,zorder=200)
cb = plt.colorbar(cs,orientation='horizontal',aspect=80, fraction=.05, pad=.3, ax=bx[4])  # draw colorbar
cb.set_label(label='Wind direction (deg.)',fontsize=15)





bx[3].set_ylabel('LKF Area', fontsize=15)
bx[3].tick_params(axis="x", labelsize=12)
bx[3].tick_params(axis="y", labelsize=12)
bx[3].set_xlim(fig_start,fig_end)

bx[3].plot(afs_dates,lkf,c='k',linestyle='None',marker='o',ms=3)

bx2 = bx[3].twinx()
bx2.set_ylabel('DE count', fontsize=15)
bx2.plot(afs_dates,dec,c='r',linestyle='None',marker='o',ms=3)





#sea ice deformation from buoys - Jenny's calculations
inpath = '../data/Buoys_JennyHutchings/'
fnames = sorted(glob(inpath+'strainrate_MOSAiC_DN_*.csv'))
outpath_events = '../data/'



for fname in fnames:
    print(fname)
    time = np.array(getColumn(fname,0))
    dt_jenny = [ datetime.strptime(time[x], "%Y-%m-%dT%H:%M:%S.%f") for x in range(len(time)) ]
    
    
    #get total deformation
    dd = getColumn(fname,4); dd = np.array(dd,dtype=np.float)*24*3600
    ss = getColumn(fname,5); ss = np.array(ss,dtype=np.float)*24*3600
    tt = np.sqrt(dd**2+ss**2)
    
    tt_cum = np.cumsum(tt)
    
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
    
    #events
    dtD = tt_cum[1:]-tt_cum[:-1]
    events_b = np.where(dtD>th,0,1)
    
    #these values depend on the starting value (they detect change!)
    #all have to start with the same value = 0, not an event    
    events_b[0]=1
    dt_events_b = events_b[1:]-events_b[:-1]

    event_start_b = np.ma.array(dt_jenny[2:],mask=dt_events_b>-1).compressed()
    event_end_b = np.ma.array(dt_jenny[2:],mask=dt_events_b<1).compressed()
    
    #if we start with not an event, we should first have a start
    if len(event_end_b) < len(event_start_b):
        event_end_b=np.append(event_end_b,event_start_b[-1])
    
    #shade each event    
    for i in range(0,len(event_start_b)):
        bx[0].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0., ymax=.5)
        bx[1].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0., ymax=.5)
        bx[2].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0., ymax=.5)
        bx[3].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0., ymax=.5)
        bx[4].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0., ymax=.5)

    
    #small scales separatelly
    if idx=='1':
        bx[0].plot(dt_jenny,tt_cum,c=color,label=label,alpha=.7)
    else:
        bx[0].plot(dt_jenny,tt_cum,c=color,label=label,alpha=.7)
    

    #write out these event starts and ends
    outfile = outpath_events+'event_start_buoys_'+idx+'.csv'
    print(outfile)

    with open(outfile, 'wb') as f:
        np.savetxt(f, event_start_b, fmt="%s", delimiter=",")

    outfile = outpath_events+'event_end_buoys_'+idx+'.csv'
    print(outfile)

    with open(outfile, 'wb') as f:
        np.savetxt(f, event_end_b, fmt="%s", delimiter=",")
    
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
    
    #pole hole
    pole_hole=tt_cum.mask
    dt_luisa = np.ma.array(dt_luisa,mask=pole_hole)

    idx=fname.split('_')[-3]
    if idx=='100km':
        label=idx+' SAR'
        color='r'
        th = .05
    elif idx=='50km':
        label=idx+' SAR'
        color='darkred'
        th=.05
    else:
        label=''
        color='.75'
        th=.05
    
    #events
    dtD = tt_cum[1:]-tt_cum[:-1]
    events_b = np.where(dtD>th,0,1)
    
    #these values depend on the starting value (they detect change!)
    #all have to start with the same value = 0, not an event    
    events_b[0]=1
    events_b[pole_hole[1:]]=1
    dt_events_b = events_b[1:]-events_b[:-1]

    event_start_b = np.ma.array(dt_luisa[2:],mask=dt_events_b>-1).compressed()
    event_end_b = np.ma.array(dt_luisa[2:],mask=dt_events_b<1).compressed()
    
    #if we start with not an event, we should first have a start
    if len(event_end_b) < len(event_start_b):
        event_end_b=np.append(event_end_b,event_start_b[-1])
    
    #shade each event    
    for i in range(0,len(event_start_b)):
        bx[0].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0.5, ymax=1)
        bx[1].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0.5, ymax=1)
        bx[2].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0.5, ymax=1)
        bx[3].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0.5, ymax=1)
        bx[4].axvspan(event_start_b[i], event_end_b[i],color=color,alpha=.4, ymin=0.5, ymax=1)
    
    
    #these are values only once per day and sums are much lower!
    bx1.plot(dt_luisa,tt_cum,c=color,label=label,alpha=.7)

bx1.legend(loc='upper left',ncol=3)
bx[0].legend(loc='lower right',ncol=3)

fig2.savefig(outpath+'event_ts.png',bbox_inches='tight')
