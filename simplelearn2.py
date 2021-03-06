import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import  datetime,timedelta,date
from dateutil.parser import parse
import os,sys,re
sys.path.append('/Users/elex-test/Documents/utilities/dataworks_utilities')
import dataworks_utilities as ut
import timeit

import fraud_judger as fj
pd.options.mode.chained_assignment = None  # default='warn'
np.random.seed(42)
start = timeit.default_timer()

path_dir='/Users/elex-test/Downloads/allnetworks/'

all_networks=pd.DataFrame()
for i in range(5,13):
    all_networks_i=pd.read_csv(os.path.join(path_dir,'all_networks2017'+str(i).zfill(2)+'.csv'))
    all_networks_i['month']=str(i).zfill(2)
    all_networks=all_networks.append(all_networks_i,ignore_index=True)

all_networks1=all_networks.drop_duplicates(keep=False)

print('Reports loaded.',timeit.default_timer()-start )

Reasons=all_networks1['Reason']

def which_type(string):
    type_A='Unmatched countries'
    type_B='This user has abnormally short click to install time'
    type_C='This user came from known bad data center or proxy IP addresses'
    type_D='This user has a bot-pattern time spent in the game'
    type_E='The user is strongly correlated with'

    if string.startswith(type_A):
        return 'type_A'
    elif string.startswith(type_B):
        return 'type_B'
    elif string.startswith(type_C):
        return 'type_C'
    elif string.startswith(type_D):
        return 'type_D'
    elif string.startswith(type_E):
        return 'type_E'
    else:
        return 'unknown_type'


Reason_types=ut.apply_by_multiprocessing(all_networks1['Reason'],which_type,workers=6)
alnet_A=all_networks1[Reason_types=='type_A']
alnet_B=all_networks1[Reason_types=='type_B']
alnet_C=all_networks1[Reason_types=='type_C']
alnet_D=all_networks1[Reason_types=='type_D']
alnet_E=all_networks1[Reason_types=='type_E']

all_networks1['reason_type']=Reason_types
print('Types parsed.',timeit.default_timer()-start)

rep_cofi_thresh=1.0
resci=all_networks1[(all_networks1['Confidence'].astype(float)>=rep_cofi_thresh)&
                    (all_networks1['reason_type']=='type_E')]
#resci07=resci[resci['month']=='07']
resci07=resci

tmp4_0=pd.read_csv(os.path.join(path_dir,'all_original.csv'), low_memory=False)

#resci07=resci[(resci['month']=='07')|(resci['month']=='08')]
#tmp4_0=tmp4_0.append(pd.read_csv(os.path.join(path_dir,'result_predict_20180208_data_new1.csv'), low_memory=False))

fill_values={'click_install_time':0, 'day_2_login_times':0, 'login_days':0,
        'day_3_7_logintimes':0, 'psum':0, 'g_paytimes':0,'totalonlinetime':0,
        'min_ip_swith_time':np.inf,'min_country_switch_time':np.inf}


tmp4_0.fillna(fill_values,inplace=True)
tmp4_0['ip_switch_max_speed']=1/tmp4_0['min_ip_swith_time']
tmp4_0['country_switch_max_speed']=1/tmp4_0['min_country_switch_time']

tmp4=tmp4_0[tmp4_0['channel']!='Organic']

tmp4a=tmp4[tmp4['adid'].isin(all_networks1['adid'])]
tmp4b=tmp4[~(tmp4['adid'].isin(all_networks1['adid']))]
#
#resci07a=resci07[resci07['adid'].isin(tmp4['adid'])]
#resci07b=resci07[~resci07['adid'].isin(tmp4['adid'])]
#
#all_networks1a=all_networks1[all_networks1['adid'].isin(tmp4['adid'])]
#all_networks1b=all_networks1[~all_networks1['adid'].isin(tmp4['adid'])]

tmp4a['marker']=1
tmp4b['marker']=0

tmp4t=tmp4a.merge(resci07[['adid','reason_type','Confidence']],on='adid')


N_num=5
tmp_some=tmp4b.sample(N_num*len(tmp4a))
#tmp_some=tmp4b

tmptotal=tmp4t.append(tmp_some,ignore_index=True)

short_time_thresh=25
short_time=fj.check_time_diff(tmptotal,short_time_thresh)['value'].astype(int).rename('short_time')

ip_file=os.path.join(path_dir,'ip_less.csv')
ip_subnet=os.path.join(path_dir,'ipsubnet.csv')
ips=pd.read_csv(ip_file,header=None,squeeze=True).tolist()
ipsubnets=pd.read_csv(ip_subnet,header=None,squeeze=True).tolist()

ip_in_lst=fj.check_ip_in_list(tmptotal,ips)['value'].astype(int).rename('ip_in_lst')
ip_in_sub=fj.check_ip_in_subnet(tmptotal,ipsubnets).astype(int).rename('ip_in_sub')

country_unmatch=fj.check_country_not_match(tmptotal)['value'].astype(int).rename('country_unmatch')

tmptotal['short_time']=short_time
tmptotal['ip_in_lst']=ip_in_lst
tmptotal['ip_in_sub']=ip_in_sub
tmptotal['country_unmatch']=country_unmatch

def strip_multiple_countries(x):
    if x is np.nan:
        return np.nan
    else:
        return len(x.split(","))

tmptotal['use_multiple_countries']=tmptotal['countriestr'].apply(strip_multiple_countries)

def supervised_count_feature(s1,s0):
    a1=s1.groupby(s1).count()
    a0=s0.groupby(s0).count()
    b0,b1=a0.align(a1)
    c1=b1.fillna(0)
    c0=b0.fillna(0)
    ss=(c1/(c0+c1))
    return ss

def supervised_add_count(ser,marker):
    ss=supervised_count_feature(ser[marker==1],ser[marker==0])
    new_ser=ser.map(ss)
    return new_ser,ss
    
tmptotal['app_num'],app_ss=supervised_add_count(tmptotal['app'],tmptotal['marker'])
tmptotal['channel_num'],channel_ss=supervised_add_count(tmptotal['channel'],tmptotal['marker'])
tmptotal['city_num'],city_ss=supervised_add_count(tmptotal['city'],tmptotal['marker'])
tmptotal['country_num'],country_ss=supervised_add_count(tmptotal['country'],tmptotal['marker'])
tmptotal['device_name_num'],device_name_ss=supervised_add_count(tmptotal['device_name'],tmptotal['marker'])
tmptotal['device_type_num'],device_type_ss=supervised_add_count(tmptotal['device_type'],tmptotal['marker'])
tmptotal['gameversion_num'],gameversion_ss=supervised_add_count(tmptotal['gameversion'],tmptotal['marker'])
tmptotal['install_iptocountry_num'],install_iptocountry_ss=supervised_add_count(tmptotal['install_iptocountry'],tmptotal['marker'])
tmptotal['ip_num'],ip_ss=supervised_add_count(tmptotal['ip'],tmptotal['marker'])
tmptotal['ipstr_num'],ipstr_ss=supervised_add_count(tmptotal['ipstr'],tmptotal['marker'])
tmptotal['isp_num'],isp_ss=supervised_add_count(tmptotal['isp'],tmptotal['marker'])
tmptotal['os_name_num'],os_name_ss=supervised_add_count(tmptotal['os_name'],tmptotal['marker'])
tmptotal['os_version_num'],os_version_ss=supervised_add_count(tmptotal['os_version'],tmptotal['marker'])


def strip_os_version(x):
    y=re.findall(r'^\d{1,2}\.\d{1,2}', x)
    if y:
        return y[0]
    else:
        y=re.findall(r'^\d{1}$', x)
        if y:
            return y[0]+'.0'
        elif x=='unknown':
                return x
        else:
            return 'other'
tmptotal['os_ver_reform']=tmptotal['os_version'].apply(strip_os_version)

def normal_version(x):
    xx=(
        ((x['os_name']=='ios')&(
                ((x['os_ver_reform']>'6')&(x['os_ver_reform']<='9.9'))|
                ((x['os_ver_reform']>'1')&(x['os_ver_reform']<='12'))
                )
    )|(
    (x['os_name']=='android')&(x['os_ver_reform']>'2')&(x['os_ver_reform']<'9')
    )|(x['os_ver_reform']=='unknown')
    )

    return xx.rename('if_ver_norm')
    
tmptotal['if_ver_norm']=normal_version(tmptotal).astype(int)

current_date=date(2017,7,31)
origin_date=date(2011,3,29)
days_delta=(current_date-origin_date).days

android_days=pd.read_csv('android_days_com.csv',dtype={'release_date':str,
                                                   'version':str,
                                                   'percentage':np.float64,
                                                   'days':np.int32})
ios_days=pd.read_csv('ios_days.csv',dtype={'release_date':str,
                                                   'version':str,
                                                   'percentage':np.float64,
                                                   'days':np.int32})
android_dict=android_days[['version','days']].set_index('version')['days'].to_dict()
ios_dict=ios_days[['version','days']].set_index('version')['days'].to_dict()
        
def version_value(x1,x2):
    if x1=='ios':
        yy=ios_dict[x2] if x2 in ios_dict.keys() else np.nan
    elif x1=='android':
        yy=android_dict[x2] if x2 in android_dict.keys() else np.nan
    else:
        yy=np.nan
    return yy   

tmptotal['os_days']=list(map(lambda x1,x2: days_delta-version_value(x1,x2), 
        tmptotal['os_name'],tmptotal['os_ver_reform']))

tmptotal['os_num']=tmptotal['os_name'].apply(lambda x: 0 if x=='ios' else 1)
    

#def strip_game_version(x):
#    y=re.match(r'^(\d{1,2})\.(\d{1,2})\.(\d{1,2})', x)
#    if y:
#        return int(y.group(1))*10000+int(y.group(2))*100+int(y.group(3))
#    else:
#        return np.nan
#
#dd1=tmptotal['gameversion'].apply(strip_game_version)
#tmptotal['num_gameversion']=dd1.max()-dd1
#####
app_days=pd.read_csv('cok_days_com.csv',dtype={'release_date':str,
                                                   'version':str,
                                                   'days':np.int32})
app_dict=app_days[['version','days']].set_index('version')['days'].to_dict()
tmptotal['app_days']=tmptotal['gameversion'].apply(lambda x: days_delta-app_dict[x]\
        if x in app_dict.keys() else np.nan)
########
tmptotal['overtime']=(tmptotal['totalonlinetime']>7*24*3600*1000).astype(int)

old_os_days=1200
tmptotal['is_old_os']=(tmptotal['os_days']>old_os_days).astype(int)

old_app_days=180
tmptotal['is_old_app']=(tmptotal['app_days']>old_app_days).astype(int)

#sel_cols=[ 'country_switch_max_speed', 'day_2_login_times', 'day_3_7_logintimes','g_paytimes',
#          'ip_switch_max_speed','login_days',  'psum','totalonlinetime', 'short_time', 'ip_in_lst',
#          'ip_in_sub','country_unmatch', 'app_num', 'channel_num', 'city_num', 
#          'country_num','device_name_num', 'device_type_num', 'gameversion_num',
#          'install_iptocountry_num', 'ip_num', 'ipstr_num', 'isp_num','os_name_num', 'os_version_num']


#sel_cols=[ 'country_switch_max_speed', 'day_2_login_times', 'day_3_7_logintimes','g_paytimes',
#          'ip_switch_max_speed','login_days',  'psum','totalonlinetime', 'short_time', 'ip_in_lst',
#          'ip_in_sub','country_unmatch','app_days','if_ver_norm','os_days','os_num','overtime',
#          'is_old_os','is_old_app','use_multiple_countries']

#sel_cols=['day_2_login_times', 'day_3_7_logintimes','ip_switch_max_speed','login_days',
#          'totalonlinetime', 'short_time', 'ip_in_lst','ip_in_sub','country_unmatch',
#          'app_days','os_days','os_num','is_old_os','is_old_app','use_multiple_countries']

sel_cols=['login_days','use_multiple_countries','ip_in_sub','country_unmatch','app_days','os_days','os_num',
          'is_old_os','is_old_app']

#sel_cols=[ 'login_days', 'ip_in_lst','ip_in_sub','country_unmatch','app_days','os_days','os_num',
#          'is_old_os','is_old_app']

#sel_cols=[ 'country_switch_max_speed', 'day_2_login_times', 'day_3_7_logintimes','g_paytimes',
#          'ip_switch_max_speed','login_days',  'psum','totalonlinetime', 'short_time', 'ip_in_lst',
#          'ip_in_sub','country_unmatch', 'app_num']




tmpselect_part0=tmptotal[sel_cols].fillna(tmptotal[sel_cols].mean())
tmpmarker_part0=tmptotal['marker']

from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(max_depth=20, n_estimators=200,n_jobs=-1,max_features='sqrt')
forest.fit(tmpselect_part0, tmpmarker_part0)
features_selected = pd.Series(forest.feature_importances_,index=tmpselect_part0.columns)
features_sorted=features_selected.sort_values(ascending=False)



from sklearn.model_selection import train_test_split
tmp_train,tmp_test=train_test_split(tmptotal,random_state=1)

tmpselect_part=tmp_train[sel_cols].fillna(tmp_train[sel_cols].mean())
tmpmarker_part=tmp_train['marker']


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

s1 = StandardScaler()
t0=s1.fit_transform(tmpselect_part)
weighs=np.ones(len(features_selected))
#weighs=features_selected.values
t1=t0*weighs
#t1=t0

n_clusters=2000
k1 = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=1000, n_init=10)
k1.fit(t1)
k_r=k1.predict(t1)
kmc=tmp_train.assign(predict=k_r)
cs=k1.cluster_centers_

kmc1=kmc[kmc['marker']==1]
kmc0=kmc[kmc['marker']==0]

sta1=kmc1['predict'].groupby(kmc1['predict']).count()
sta0=kmc0['predict'].groupby(kmc0['predict']).count()
sta1a=sta1/len(kmc1)
sta0a=sta0/len(kmc0)
dif1=sta1a-sta0a
dif2=sta1a/sta0a
result=pd.concat([sta1,sta0,sta1a,sta0a,dif1,dif2],axis=1)
result.columns=['sta1','sta0','sta1a','sta0a','dif1','dif2']
resee=result.sort_values('dif2',ascending=False)
resee['cumsta1a']=np.cumsum(resee['sta1a'])
resee['cumsta0a']=np.cumsum(resee['sta0a'])
resee['cumsta1']=np.cumsum(resee['sta1'])
resee['cumsta0']=np.cumsum(resee['sta0'])
resee['dif3']=resee['sta1']/resee['sta0']

resee.fillna(0,inplace=True)

pick_thresh=0.4
pick_clusters=resee[resee['dif2']>=pick_thresh].sort_values(by='dif2',
                   ascending=False).index.tolist()

#pick_number=144
#pick_clusters=resee['dif2'].sort_values(ascending=False).index.tolist()[0:pick_number]


from sklearn.metrics.pairwise import paired_distances  
kmc['dis']=pd.Series(map(lambda x1,x2: paired_distances(x1.reshape(1,-1),cs[x2].reshape(1,-1))[0],
        t1,k_r))

#radius=kmc['dis'].groupby(kmc['predict']).mean()

kmcc=kmc[kmc['marker']==1]
radius=kmcc['dis'].groupby(kmcc['predict']).max()
ss=resee
tt=pd.merge(ss,pd.DataFrame(radius),left_index=True, right_index=True)

cut_len=1.0
radius_cut=radius*cut_len

#kmc['rectified']=(kmc['dis']<kmc['predict'].map(radius_cut))&(
#        kmc['predict'].isin(pick_clusters)).astype(int)
 
kmc['rectified']=kmc['predict'].isin(pick_clusters).astype(int)

kmc['dis0']=kmc['predict'].map(tt['dis'])
kmct=kmc[kmc['predict'].isin(pick_clusters)]
kmct1=kmct[kmct['dis']>kmct['dis0']]
   
comp=kmc[['marker','rectified']]    
compa=(comp['marker']==comp['rectified'])

print(comp['marker'].groupby(comp['marker']).count())
print(comp['rectified'].groupby(comp['rectified']).count())
print(comp[(comp['marker']==1)&(comp['rectified']==1)].shape)

#kmc07=kmc[kmc['month']=='07']
#kmc08=kmc[kmc['month']=='08']
#
#print('7')
#print(kmc07['marker'].groupby(kmc07['marker']).count())
#print(kmc07['rectified'].groupby(kmc07['rectified']).count())
#
#print('8')
#print(kmc08['marker'].groupby(kmc08['marker']).count())
#print(kmc08['rectified'].groupby(kmc08['rectified']).count())

#from sklearn.metrics import classification_report
#print(classification_report(comp['marker'],comp['rectified']))


tmpselect_part2=tmp_test[sel_cols].fillna(tmp_test[sel_cols].mean())
tmpmarker_part2=tmp_test['marker']

tmpselect_part2_st0=s1.transform(tmpselect_part2)
tmpselect_part2_st=tmpselect_part2_st0*weighs


tmpselect_part2_res=k1.predict(tmpselect_part2_st)
tmpcluster=tmp_test.assign(predict=tmpselect_part2_res)

tmpcluster['dis']=pd.Series(map(lambda x1,x2: paired_distances(x1.reshape(1,-1),cs[x2].reshape(1,-1))[0],
        tmpselect_part2_st,tmpselect_part2_res))

pick_clusters1=pick_clusters
#pick_clusters1=resee.index.tolist()
tmpcluster['rectified_1']=tmpcluster['predict'].isin(pick_clusters1).astype(int)

tmpcluster['dis0']=tmpcluster['predict'].map(tt['dis'])

tmpclustert=tmpcluster[tmpcluster['predict'].isin(pick_clusters)]
tmpclustert1=tmpclustert[tmpclustert['dis']>tmpclustert['dis0']]


print(tmpcluster['rectified_1'].groupby(tmpcluster['rectified_1']).count())

tmpcluster_res=tmpcluster['adid'][tmpcluster['rectified_1']==1]

from sklearn.metrics import classification_report

print(classification_report(tmpcluster['rectified_1'],tmpcluster['marker']))






end=timeit.default_timer()
print(end-start)  

#start=timeit.default_timer()
#from sklearn.cluster import DBSCAN
#
#db = DBSCAN(eps=0.3, min_samples=10,n_jobs=-1).fit(t1)
#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
#core_samples_mask[db.core_sample_indices_] = True
#labels = db.labels_
#
## Number of clusters in labels, ignoring noise if present.
#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

#import hdbscan
#clusterer = hdbscan.HDBSCAN(min_cluster_size=200, prediction_data=True).fit(t1)
##clusterer.fit(t1)
#
#clse=pd.Series(clusterer.labels_)
#print(clse.groupby(clse).count().sort_values(ascending=False))
#
#end=timeit.default_timer()
#print(end-start)  


