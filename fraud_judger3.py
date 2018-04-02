#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:25:09 2018

@author: elex-test
"""


import sys,os,re
import numpy as np
import pandas as pd
import ipaddress as ip
from datetime import date
from dateutil.parser import parse
from sklearn.externals import joblib
pd.options.mode.chained_assignment = None  # default='warn'


#path

MODEL_FILES_PATH='.'


#files

IP_LIST='ip_less.csv'
IP_SUBNET='ipsubnet.csv'

ANDROID_VER='android_days_com.csv'
IOS_VER='ios_days.csv'

APP_VER='cok_days_com.csv'

FEATURES='features_selected.csv'
DETAILS='ss_com.csv'
ST_SC='s2'
CLUS='k2'


#variables

time_threshold=25
old_os_days=1200
old_app_days=180
pick_thresh=0.4
#pick_number=144


#load files

ips=pd.read_csv(os.path.join(MODEL_FILES_PATH,IP_LIST),
                header=None,squeeze=True).tolist()
ipsubnets=pd.read_csv(os.path.join(MODEL_FILES_PATH,IP_SUBNET),
                      header=None,squeeze=True).tolist()

android_days=pd.read_csv(os.path.join(MODEL_FILES_PATH,ANDROID_VER),
                         dtype={'release_date':str,'version':str,
                                'percentage':np.float64,'days':np.int32})
ios_days=pd.read_csv(os.path.join(MODEL_FILES_PATH,IOS_VER),
                     dtype={'release_date':str,'version':str,
                            'percentage':np.float64,'days':np.int32})

app_days=pd.read_csv(os.path.join(MODEL_FILES_PATH,APP_VER),
                     dtype={'release_date':str,'version':str,'days':np.int32})

features_selected=pd.read_csv(os.path.join(MODEL_FILES_PATH,FEATURES))
ss=pd.read_csv(os.path.join(MODEL_FILES_PATH,DETAILS)).set_index('predict')
s1=joblib.load(os.path.join(MODEL_FILES_PATH,ST_SC))
k1=joblib.load(os.path.join(MODEL_FILES_PATH,CLUS))


#constants

type_A='Unmatched countries: targeted country and ip country are different.'
type_B='This user has abnormally short click to install time.'
type_C='This user came from known bad data center or proxy IP addresses.'
type_D='This user has a bot-pattern time spent in the game.'
type_E='The user is strongly correlated with '
E_tail=(' other fraudulent users. Their top correlated features and the'+
        ' percentages of the users sharing the same features are as follows: ')
type_F="The user's os version is not in the version list."

origin_date=date(2011,3,29)

sel_cols=['login_days','use_multiple_countries','ip_in_sub','country_unmatch',
          'app_days','os_days','os_num','is_old_os','is_old_app']
report_content=['channel', 'gaid', 'adid', 'ip', 'country', 'city', 'isp',
                'click_referer', 'clicktime', 'installtime', 'app', 'gameversion',
                'device_name', 'device_type', 'os_name', 'os_version','ipstr', 
                'countriestr', 'day_2_login_times','login_days', 'day_3_7_logintimes',
                'psum', 'g_paytimes','install_iptocountry', 'min_ip_swith_time',
                'min_country_switch_time','totalonlinetime']

#related quantities

android_dict=android_days[['version','days']].set_index('version')['days'].to_dict()
ios_dict=ios_days[['version','days']].set_index('version')['days'].to_dict()
app_dict=app_days[['version','days']].set_index('version')['days'].to_dict()
pick_clusters=ss[ss['dif2']>=pick_thresh].sort_values(by='dif2',
                   ascending=False).index.tolist()
#pick_clusters=ss['dif2'].sort_values(ascending=False).index.tolist()[0:pick_number]


#functions

#check country unmatch.type_A
def country_match(x,xlst):
    if pd.isnull(x) or pd.isnull(xlst):
        return True
    elif x.lower() in xlst.lower().split(','):
        return True
    else:
        return False    
def check_country_not_match(data):
    value=pd.Series(index=data.index,name='value')
    for i in data.index.tolist():
        value[i]=not country_match(data.loc[i,'install_iptocountry'],
                       data.loc[i,'countriestr'])
    reason=pd.Series(index=data.index,name='reason')
    reason[value]=type_A
    return pd.concat([value,reason],axis=1)  

#check time difference.type_B
def check_time_diff(data,time_threshold):        
    value=pd.Series((data['click_install_time']<time_threshold)
                     &(data['click_install_time']>=0),
                    index=data.index,name='value')
    reason=pd.Series(index=data.index,name='reason')
    reason[value]=type_B
    return pd.concat([value,reason],axis=1)

#check if ip in suspicious ip list or subnet.type_C
def check_ip_in_list(data,ips):
    value=data['ip'].isin(ips).rename('value')
    reason=pd.Series(index=data.index,name='reason')
    reason[value]=type_C
    return pd.concat([value,reason],axis=1)

#check if version is normal.type_F
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
    return xx
def not_normal_version(x):
    value=~normal_version(x).rename('value')    
    reason=pd.Series(index=x.index,name='reason')
    reason[value]=type_F
    return pd.concat([value,reason],axis=1)

#helper functions to check if the ip is in suspicious subnet.
def ip_in_subnet(ip,ips_obj):
    return True in set(map(lambda x: ip in x if ip is not None else False, ips_obj))
def check_ip_in_subnet(data,ip_subnet):    
    ipstrlst=data['ip'].astype(str).tolist()
    theip=[ip.IPv4Address(i) if i!='nan' else None for i in ipstrlst ]
    ips_obj=[ip.IPv4Network(i+'/20') for i in ip_subnet]      
    res=pd.Series([ip_in_subnet(i,ips_obj) for i in theip],
                   name='IPs are in the /20 IP subnet')
    return res

#helper function to check how many countries the user has ever correlated.
def strip_multiple_countries(x):
    if x is np.nan:
        return np.nan
    else:
        return len(x.split(","))
    
#helper functions to handle os versions.    
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
def version_value(x1,x2):
    if x1=='ios':
        yy=ios_dict[x2] if x2 in ios_dict.keys() else np.nan
    elif x1=='android':
        yy=android_dict[x2] if x2 in android_dict.keys() else np.nan
    else:
        yy=np.nan
    return yy  

#prepare the data to only consider non-organic users.
def data_preprocessor(tmp):
    tmp=tmp[tmp['channel']!='Organic']
    
    fill_values={'click_install_time':0, 'day_2_login_times':0, 'login_days':0,
        'day_3_7_logintimes':0, 'psum':0, 'g_paytimes':0,'totalonlinetime':0,
        'min_ip_swith_time':tmp['min_ip_swith_time'].mean(),
        'min_country_switch_time':tmp['min_country_switch_time'].mean()}
    tmp.fillna(fill_values,inplace=True)   
    
    return tmp

#clustering and combine the components of the result.
def belongs_to_clusters(tmp,current_date):
    
    #tmp['ip_switch_max_speed']=1/tmp['min_ip_swith_time']
    #tmp['country_switch_max_speed']=1/tmp['min_country_switch_time']
    
    if_A=check_country_not_match(tmp)
    country_unmatch=if_A['value'].astype(int).rename('country_unmatch')
    
    if_B=check_time_diff(tmp,time_threshold)
    short_time=if_B['value'].astype(int).rename('short_time')
    
    if_C=check_ip_in_list(tmp,ips)
    ip_in_lst=if_C['value'].astype(int).rename('ip_in_lst')
    
    tmp['os_ver_reform']=tmp['os_version'].astype(str).apply(strip_os_version)
    if_F=not_normal_version(tmp)
    if_ver_abnorm=if_F['value'].astype(int).rename('if_ver_abnorm')
    
    tmp['country_unmatch']=country_unmatch
    tmp['short_time']=short_time
    tmp['ip_in_lst']=ip_in_lst
    tmp['if_ver_abnorm']=if_ver_abnorm
    
    tmp['ip_in_sub']=check_ip_in_subnet(tmp,ipsubnets).astype(int).rename('ip_in_sub')
    tmp['use_multiple_countries']=tmp['countriestr'].apply(strip_multiple_countries)
        
    days_delta=(parse(current_date).date()-origin_date).days
    tmp['os_days']=list(map(lambda x1,x2: days_delta-version_value(x1,x2), 
            tmp['os_name'],tmp['os_ver_reform']))
    
    tmp['os_num']=tmp['os_name'].apply(lambda x: 0 if x=='ios' else 1)
    
    tmp['app_days']=tmp['gameversion'].apply(lambda x: days_delta-app_dict[x]\
            if x in app_dict.keys() else np.nan)
    #tmp['overtime']=(tmp['totalonlinetime']>7*24*3600*1000).astype(int)
    
    tmp['is_old_os']=(tmp['os_days']>old_os_days).astype(int)
    tmp['is_old_app']=(tmp['app_days']>old_app_days).astype(int)
    
    
    tmp_s=tmp[sel_cols].fillna(tmp[sel_cols].mean())
    
    tmp_a=s1.transform(tmp_s)
    
    k_res=k1.predict(tmp_a)
    kmcluster=tmp.assign(predict=k_res)
    
    cluster_value=(kmcluster['predict'].isin(pick_clusters)).rename('value')
    cluster_reason=pd.Series(index=cluster_value.index,name='reason')
    cluster_reason[cluster_value]=kmcluster['predict'].map(
            type_E+ss['sta1'].astype(int).astype(str)+ E_tail+ss['reason'])
    if_E=pd.concat([cluster_value,cluster_reason],axis=1)    
    
    return(if_A,if_B,if_C,if_F,if_E,kmcluster)
    
#organize all the results according to all types.
def organize_result(if_A,if_B,if_C,if_F,if_E,kmcluster):
    
    value=(if_A['value']|if_B['value']|if_C['value']|
            if_F['value']|if_E['value'])
    
    reason=(if_A['reason'].fillna(if_B['reason']).fillna(if_C['reason']).
            fillna(if_F['reason']).fillna(if_E['reason']))
    
    result=pd.concat([kmcluster[report_content],value,reason],axis=1)
    
    ret=result[result['value']].drop('value',axis=1)
    
    print('oo1')
    print(len(ret),len(result),len(ret)/len(result))
    
    return ret


#For validation
    
RECORD_FILE='report09.csv'
resci=pd.read_csv(os.path.join(MODEL_FILES_PATH,RECORD_FILE),low_memory=False)
def check_result(if_A,if_B,if_C,if_F,if_E,kmcluster,resci):
    kmcluster['rectified']=if_E['value'].astype(int)    
    print(kmcluster['rectified'].groupby(kmcluster['rectified']).count())    
    tmpres=kmcluster['adid'][kmcluster['rectified']==1]   
    inter=np.intersect1d(tmpres,resci['adid'])
    print(len(inter))
    print(len(inter)/len(tmpres))
    aa=kmcluster['adid'].astype(str)
    bb=resci['adid'].astype(str)
    aabb=np.intersect1d(aa,bb)
    print('oo2')
    print(len(aa),len(bb),len(aabb))
    cca=kmcluster['adid'][if_A['value']].astype(str)
    ccb=kmcluster['adid'][if_B['value']].astype(str)
    ccc=kmcluster['adid'][if_C['value']].astype(str)
    ccf=kmcluster['adid'][if_F['value']].astype(str)
    cce=kmcluster['adid'][if_E['value']].astype(str)
    print(len(cca),len(ccb),len(ccc),len(ccf),len(cce))
    print(len(np.intersect1d(cca,bb)),len(np.intersect1d(ccb,bb)),len(np.intersect1d(ccc,bb)),
          len(np.intersect1d(ccf,bb)),len(np.intersect1d(cce,bb)))

   
def fraud_detector(current_date, input_file, output_file):
    
    #Read files
    data0=pd.read_csv(input_file,low_memory=False)
    
    #Data preparation
    data=data_preprocessor(data0)
            
    if_A,if_B,if_C,if_F,if_E,kmcluster=belongs_to_clusters(data,current_date)    
   
    #Organize the results and write it to output file.
    result=organize_result(if_A,if_B,if_C,if_F,if_E,kmcluster)
    result.to_csv(output_file,index=False)
    
    check_result(if_A,if_B,if_C,if_F,if_E,kmcluster,resci)
        
    return result
    
                   

if __name__ == "__main__":
    """
    sys.argv[1]: Current date.
    sys.argv[2]: Input csv file.
    sys.argv[3]: Output file.
    """
    
#    result=fraud_detector(sys.argv[1],sys.argv[2],sys.argv[3])
    
    current_date='2017-10-01'
    input_file=os.path.join(MODEL_FILES_PATH,'original_data20171001.csv')
    output_file=os.path.join(MODEL_FILES_PATH,'checker_20171001.csv')
    
    import timeit
    start = timeit.default_timer()
    result=fraud_detector(current_date, input_file, output_file)
    end=timeit.default_timer()
    print(end-start) 
    