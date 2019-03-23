#thomas feiring model 
import math
import numpy as np
import pandas as pd
#enter the year for which you need prediction starting 2019
year=2019
number_of_days=366*2
day=0
df = pd.read_csv('../LSTM/groundtruth.csv')
u=df['Mean']
X_t= u[0]
sd=df['St dev']
print("Month,Year,Inflow")
#lag -1 correlation
lag=df['co relation']
np.random.seed(9001)
for i in range(number_of_days):
    rn=np.random.normal(0,1,1)[0]
    z_t=(X_t-u[day])/sd[day]
    z_t1=lag[day]*z_t+rn*math.sqrt(1-lag[day]*lag[day])
    X_t1=u[(day+1)%12]+z_t1*sd[(day+1)%12]
    print(day,",",year,",",X_t1)
    if(day==365):
        year=year+1
    day=(day+1)%366
    X_t=X_t1
