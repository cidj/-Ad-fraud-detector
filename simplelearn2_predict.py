#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:17:26 2018

@author: elex-test
"""

tmp5_0=pd.read_csv(os.path.join(path_dir,'result_predict_20180208_data_new1.csv'), low_memory=False)

tmp5_0.fillna(fill_values,inplace=True)
tmp5_0['ip_switch_max_speed']=1/tmp5_0['min_ip_swith_time']
tmp5_0['country_switch_max_speed']=1/tmp5_0['min_country_switch_time']

index_to_drop=tmp5_0['ip'][tmp5_0['ip']=='{ip_address}'].index
tmp5=tmp5_0[tmp5_0['channel']!='Organic'].drop(index_to_drop)

short_time5=fj.check_time_diff(tmp5,short_time_thresh)['value'].astype(int).rename('short_time')

ip_in_lst5=fj.check_ip_in_list(tmp5,ips)['value'].astype(int).rename('ip_in_lst')
ip_in_sub5=fj.check_ip_in_subnet(tmp5,ipsubnets).astype(int).rename('ip_in_sub')

country_unmatch5=fj.check_country_not_match(tmp5)['value'].astype(int).rename('country_unmatch')

tmp5['short_time']=short_time5
tmp5['ip_in_lst']=ip_in_lst5
tmp5['ip_in_sub']=ip_in_sub5
tmp5['country_unmatch']=country_unmatch5

tmp5['use_multiple_countries']=tmp5['countriestr'].apply(strip_multiple_countries)

tmp5['app_num']=tmp5['app'].map(app_ss)
tmp5['channel_num']=tmp5['channel'].map(channel_ss)
tmp5['city_num']=tmp5['city'].map(city_ss)
tmp5['country_num']=tmp5['country'].map(country_ss)
tmp5['device_name_num']=tmp5['device_name'].map(device_name_ss)
tmp5['device_type_num']=tmp5['device_type'].map(device_type_ss)
tmp5['gameversion_num']=tmp5['gameversion'].map(gameversion_ss)
tmp5['install_iptocountry_num']=tmp5['install_iptocountry'].map(install_iptocountry_ss)
tmp5['ip_num']=tmp5['ip'].map(ip_ss)
tmp5['ipstr_num']=tmp5['ipstr'].map(ipstr_ss)
tmp5['isp_num']=tmp5['isp'].map(isp_ss)
tmp5['os_name_num']=tmp5['os_name'].map(os_name_ss)
tmp5['os_version_num']=tmp5['os_version'].map(os_version_ss)

tmp5['os_ver_reform']=tmp5['os_version'].apply(strip_os_version)
tmp5['if_ver_norm']=normal_version(tmp5).astype(int)

current_date1=date(2017,8,31)

days_delta1=(current_date1-origin_date).days

tmp5['os_days']=list(map(lambda x1,x2: days_delta1-version_value(x1,x2), 
        tmp5['os_name'],tmp5['os_ver_reform']))

tmp5['os_num']=tmp5['os_name'].apply(lambda x: 0 if x=='ios' else 1)

#dd1_5=tmp5['gameversion'].apply(strip_game_version)
#tmp5['num_gameversion']=dd1_5.max()-dd1_5

tmp5['app_days']=tmp5['gameversion'].apply(lambda x: days_delta1-app_dict[x]\
        if x in app_dict.keys() else np.nan)
########
tmp5['overtime']=(tmp5['totalonlinetime']>7*24*3600*1000).astype(int)
tmp5['is_old_os']=(tmp5['os_days']>old_os_days).astype(int)
tmp5['is_old_app']=(tmp5['app_days']>old_app_days).astype(int)

#from sklearn.externals import joblib
#xxxx=joblib.load('k1')
#k1=xxxx

#np.random.seed(1)


tmp5select_part=tmp5[sel_cols].fillna(tmp5[sel_cols].mean())

tmp5a0=s1.transform(tmp5select_part)

tmp5a=tmp5a0

k_res5=k1.predict(tmp5a)
kmcluster5=tmp5.assign(predict=k_res5)

kmcluster5['dis']=pd.Series(map(lambda x1,x2: paired_distances(x1.reshape(1,-1),cs[x2].reshape(1,-1))[0],
        tmp5a,k_res5))

pick_clusters1=pick_clusters
#pick_clusters1=resee.index.tolist()
kmcluster5['rectified_1']=kmcluster5['predict'].isin(pick_clusters1).astype(int)

kmcluster5['dis0']=kmcluster5['predict'].map(tt['dis'])
kmcluster5t=kmcluster5[kmcluster5['predict'].isin(pick_clusters)]
kmcluster5t1=kmcluster5t[kmcluster5t['dis']>kmcluster5t['dis0']]


print(kmcluster5['rectified_1'].groupby(kmcluster5['rectified_1']).count())

tmp5res=kmcluster5['adid'][kmcluster5['rectified_1']==1]
resci08=resci[resci['month']=='08']
resci08a=all_networks1[all_networks1['month']=='08']

inter=np.intersect1d(tmp5res,resci08['adid'])
print(len(inter))
print(len(inter)/len(tmp5res))

#select_cluster_num=resee[resee['dif2']>3].index
#re1=kmcluster5[kmcluster5['predict'].isin(select_cluster_num)].drop('predict',axis=1)
#re1['adid'].to_csv('re2.csv',index=False)