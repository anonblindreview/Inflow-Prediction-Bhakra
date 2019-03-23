import os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('../data1.csv')
df=df.values
#time series vs inflow and discharge graph
sns.set_style('darkgrid')
plt.figure()
ax1=plt.subplot(2,1,1)
line1,=ax1.plot(df[:,0],df[:,2],label='inflow')
ax2 = ax1.twinx()
line2,=ax2.plot(df[:,0],df[:,3],label='discharge',color='#ff7f0eff')
ax1.set_xlabel('Time Series')
ax1.set_ylabel('Inflow')
ax2.set_ylabel('Discharge')
#plt.title('',fontsize=14)
#fig.tight_layout()


plt.subplot(2,1,2)
line3,=plt.plot(df[:,0],df[:,1],label='Reservoir Levels(ft)')
plt.xlabel('Time Series')
plt.ylabel('Reservoir Levels(ft)')

ax1.legend((line1,line2),('Inflow', 'Discharge'))
#ax2.legend()
plt.show()

#g = sns.tsplot(x="Time",y="value",kind="line",data=df)
#g.fig.autofmt_xdate()
