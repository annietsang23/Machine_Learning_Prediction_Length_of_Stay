#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime
import matplotlib as plotlib
import plotly.graph_objs as graph



# In[2]:

def patient_activities_df(df_pa,iso=False):
    #Preliminary data analysis
    #read the patient activities data
    df_p_act=df_pa

    df_p_act.drop_duplicates(inplace=True)

    df_p_act.columns = [col.replace(' ', '') for col in df_p_act.columns]

    # Drop rows which contain null values across all columns
    df_p_act.dropna(how='all',inplace=True)

    # remove entries related to 'transfer' within the same ward
    df_p_act.drop(df_p_act[df_p_act.From_WardId==df_p_act.WardId].index,axis=0,inplace=True)
    if not iso:
        df_p_act['EventOccured']=pd.to_datetime(df_p_act['EventOccured'],format='%d/%m/%Y %H:%M')
    else:
        df_p_act['EventOccured'] = pd.to_datetime(df_p_act['EventOccured'], format='%Y-%m-%dT%H:%M:%S.%fZ')

    df_p_act.reset_index(inplace=True)

    return df_p_act

def plot_activities_over_time(df_p_act,ward=None):
    print(ward)

    timeseries={}
    #plot activities over time for all ward
    if ward ==None:
        x='All'
        for i in ('Admission','Transfer','Discharge'):
            timeseries[i]=df_p_act[(df_p_act.EventType==i)].EventOccured.dt.to_period('M').value_counts().sort_index()

    else:
        x=ward
        for i in ('Admission', 'Transfer', 'Discharge'):
            timeseries[i]=df_p_act[(df_p_act.EventType==i) &
            (df_p_act.WardName.isin(ward))].EventOccured.dt.to_period('M').value_counts().sort_index()


    return dcc.Graph(
        className='graph',
        figure={
            'data': [
                graph.Scatter(
                    x=timeseries[i].index.astype(str),
                    y=timeseries[i].values,

                    mode='lines+markers',
                    opacity=0.7,
                    name=str(i)
                ) for i in timeseries.keys()
            ],
            'layout': graph.Layout(
                title={'text':f'Number of admission/transfer/discharge cases across months ({x})'},
                xaxis={'title': f'Month'},
                yaxis={'title': 'Number of Episode References'},
            )
        })



def combine_df(df_p_act,df_wt,df_pd):

    #Prepare the label column (y) i.e. length of stays
    admission_date=df_p_act[df_p_act.EventType=='Admission'][['HospitalNumber','EpisodeReference','WardId','EventOccured']]
    admission_date.rename(columns={'EventOccured':'Admission_Date','WardId':'Admission_WardId'},inplace=True)
    transfer_date=df_p_act[df_p_act.EventType=='Transfer'][['EpisodeReference','EventOccured','WardId','From_WardId','TransferType']]
    transfer_date.rename(columns={'EventOccured':'Transfer_Date','WardId':'To_Ward'},inplace=True)
    discharge_date=df_p_act[df_p_act.EventType=='Discharge'][['EpisodeReference','EventOccured']]
    discharge_date.rename(columns={'EventOccured':'Discharge_Date'},inplace=True)

    for i in [admission_date,transfer_date,discharge_date]:
        i.drop_duplicates(inplace=True)

    discharge_date.sort_values(by='Discharge_Date',ascending=False,inplace=True)

    # Keep the most recent discharge date for each episode reference
    discharge_date=discharge_date.groupby('EpisodeReference').first()

    admission_date.sort_values(by='Admission_Date',ascending=True,inplace=True)

    # Keep the earliest admission date for each episode reference
    admission_date=admission_date.groupby('EpisodeReference').first()


    df_los=pd.merge(admission_date,transfer_date,on='EpisodeReference',how='outer')

    df_los_t=pd.merge(df_los,discharge_date,on='EpisodeReference',how='outer')

    df_los_t['Start_Date']=''
    df_los_t['End_Date']=''
    df_los_t['Ward']=''

    df_los_a=df_los_t[np.isnat(df_los_t.Transfer_Date)]

    df_los_a.Start_Date=df_los_a.Admission_Date
    df_los_a.End_Date=df_los_a.Discharge_Date
    df_los_a.Ward=df_los_a.Admission_WardId

    df_los_a=df_los_a[~(df_los_a.End_Date.isnull() | df_los_a.Start_Date.isnull())]

    df_los_b=df_los_t[~(np.isnat(df_los_t.Transfer_Date))]

    df_los_b1=df_los_b.copy()

    df_los_b2=pd.concat([df_los_b,df_los_b1],axis=0)

    df_los_b2.sort_values(by=['EpisodeReference','Transfer_Date'],inplace=True)

    df_los_b2.reset_index(inplace=True)

    df_los_b2.drop(['index'],axis=1,inplace=True)

    from numpy import nan
    for i in df_los_b2.EpisodeReference.unique():
        idx=df_los_b2[df_los_b2.EpisodeReference==i].index
        for j in range(len(idx)):
            if j==0:
                df_los_b2.loc[idx[j],'Start_Date']=df_los_b2.loc[idx[j],'Admission_Date']
                df_los_b2.loc[idx[j],'End_Date']=df_los_b2.loc[idx[j],'Transfer_Date']
                df_los_b2.loc[idx[j],'Ward']=df_los_b2.loc[idx[j],'Admission_WardId']
            elif (j+1)==len(idx):
                df_los_b2.loc[idx[j],'Start_Date']=df_los_b2.loc[idx[j],'Transfer_Date']
                df_los_b2.loc[idx[j],'End_Date']=df_los_b2.loc[idx[j],'Discharge_Date']
                df_los_b2.loc[idx[j],'Ward']=df_los_b2.loc[idx[j],'To_Ward']
            else:
                df_los_b2.loc[idx[j],'Start_Date']=df_los_b2.loc[idx[j],'Transfer_Date']
                df_los_b2.loc[idx[j],'Ward']=df_los_b2.loc[idx[j],'To_Ward']
                #end date
                boolean=(df_los_b2.loc[idx[j],'Transfer_Date']<=df_los_b2.loc[idx[j+1],'Transfer_Date'])
                if ((df_los_b2.loc[idx[j],'To_Ward']==df_los_b2.loc[idx[j+1],'From_WardId']) & boolean):
                    df_los_b2.loc[idx[j],'End_Date']=df_los_b2.loc[idx[j+1],'Transfer_Date']
                else:
                    df_los_b2.loc[idx[j],'End_Date']=nan

    df_los_b=df_los_b2[~(df_los_b2.End_Date.isnull() | df_los_b2.Start_Date.isnull())]

    df_los_b.drop_duplicates(inplace=True)

    df_p_act=pd.concat([df_los_a,df_los_b],axis=0)

    df_p_act.drop(['Admission_WardId','Admission_Date','Transfer_Date','To_Ward','Discharge_Date'],axis=1,inplace=True)

    #Calculate the length of stays
    df_p_act['los_days']=(df_p_act['End_Date']-df_p_act['Start_Date']).dt.days

    df_p_act['start_month']=df_p_act['Start_Date'].dt.month

    df_p_act.isnull().sum()

    df_p_act=df_p_act[~(df_p_act.HospitalNumber.isnull())]


    #read ward information
    df_w=df_wt

    df_w.drop(['Unnamed: 0'],axis=1,inplace=True)

    df_w.columns=df_w.iloc[0]

    df_w.drop([0],inplace=True)

    df_w=df_w[['Ward Code','Ward Name','Ward Service Type','Ward Location']]

    #convert From_WardId field to ward name based on ward type table
    def conv(x):
        if not (x == 0):
            x = str(x).strip()
            y = df_w[df_w['Ward Code'] == str(x)]['Ward Name']
            if (y is None) or str(y.values)[2:-2].strip() == "":
                return 0
            else:
                return str(y.values)[2:-2].strip()
        else:
            return 0

    df_p_act['From_WardId'] = df_p_act['From_WardId'].apply(conv)

    #combine ward type and patient activities dataframe
    df_los=pd.merge(df_p_act,df_w,left_on='Ward',right_on='Ward Code',how='inner')

    # df_p_act['Ward']=df_p_act['Ward Name']
    # df_p_act.drop(['Ward Name'],axis=1,inplace=True)

    df_los.fillna(0,inplace=True)

    #read the demographics data
    df_p_demo=df_pd
    #these fields are removed due to high percentage of null values /imbalance of classes
    df_p_demo.drop(['FOD','AdvancedDecision','SexualOrientation'],axis=1,inplace=True)

    #combine the demographic and admission fields into a new dataframe
    df_los_demo=pd.merge(df_los,df_p_demo,on='HospitalNumber')

    df_los_demo.drop_duplicates(inplace=True)

    df_los_demo.columns = [col.replace(' ', '') for col in df_los_demo.columns]

    #remove outliers which are 2 standard deviations away from mean
    def remove_outliers(data,col):
        from scipy import stats
        z_score=np.abs(stats.zscore(data[col]))
        data=data[z_score<2]
        data=data[data[col]>0]
        return data

    df_los_p=remove_outliers(df_los_demo,'los_days')
    #
    # #Save the processed dataframe as csv
    # df_los_p.to_csv('df_los_p_o.csv',index=False)

    return df_los_p
