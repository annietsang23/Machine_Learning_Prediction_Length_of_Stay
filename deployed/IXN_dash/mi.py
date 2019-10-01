import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as graph
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
import os


#Encode categorical data into label
# encoder={'label': {
#             'Gender': {'M': 1, 'F': 2},
#            'Ward Service Type': {'Adult Acute': 1,
#                                  'Adult PICU': 2,
#                                  'Older Adult Acute': 3,
#                                  'Medium Secure': 4,
#                                  'Rehabilitation': 5,
#                                  'Perinatal': 6,
#                                  'Eating Disorders': 7,
#                                  'Neuropsychiatry': 8,
#                                  'Deaf': 9},
#            'Ward Location': {'The Zinnia Centre': 1,
#                              'Highcroft Site': 2,
#                              'Eden Unit': 3,
#                              'Endeavour House': 4,
#                              'Oleaster': 5,
#                              'Mary Seacole House': 6,
#                              'Reservoir Court': 7,
#                              'Tamarind Centre': 8,
#                              'Hertford House': 9,
#                              'Ardenleigh': 10,
#                              'Ross House': 11,
#                              'Dan Mooney House': 12,
#                              'Juniper Centre': 13,
#                              'Forward House': 14,
#                              'Endeavour Court': 15,
#                              'David Bromley House': 16,
#                              'Reaside Clinic': 17,
#                              'Grove Avenue': 18,
#                              'Newbridge House': 19,
#                              'The Barberry': 20},
#            'Ward': {'Barberry - Jasmine': 1,
#                     'Ross House': 2,
#                     'Oleaster - Tazetta': 3,
#                     'Reservoir Court': 4,
#                     'Endeavour House': 5,
#                     'Juniper - Bergamot': 6,
#                     'Ardenleigh - Tourmaline': 7,
#                     0: 0,
#                     'Grove Avenue': 9,
#                     'Newbridge House': 10,
#                     'Dan Mooney House': 11,
#                     'Oleaster - Caffra': 12,
#                     'Mary Seacole House - Ward 2': 13,
#                     'Tamarind - Acacia': 14,
#                     'Mary Seacole House - Meadowcroft': 15,
#                     'Ardenleigh - Coral': 16,
#                     'Oleaster - Magnolia': 17,
#                     'Tamarind - Myrtle': 18,
#                     'Juniper - Sage': 19,
#                     'David Bromley House': 20,
#                     'Ardenleigh - Citrine': 21,
#                     'Tamarind - Sycamore': 22,
#                     'Barberry - Cilantro': 23,
#                     'Zinnia - Lavender': 24,
#                     'Rookery Gardens': 25,
#                     'Tamarind - Hibiscus': 26,
#                     'Hertford House': 27,
#                     'Eden - Female PICU': 28,
#                     'Reaside - Severn': 29,
#                     'Endeavour Court': 30,
#                     'Eden - Acute': 31,
#                     'Juniper - Rosemary': 32,
#                     'Reaside - Blythe': 33,
#                     'Reaside - Kennet': 34,
#                     'Oleaster - Melissa': 35,
#                     'Forward House': 36,
#                     'Oleaster - Japonica': 37,
#                     'Endeavour House Community Rehab': 38,
#                     'Barberry - Vitivier': 39,
#                     'Reaside - Avon': 40,
#                     'Zinnia - Saffron': 41,
#                     'Highcroft - George': 42,
#                     'Mary Seacole House - Ward 1': 43,
#                     'Barberry - Chamomile': 44}},
#  'age': [18.0, 29.0, 37.0, 45.0, 54.0, 64.0, 76.0, 96.0]}

# Unpickle encoder and models and import csv files as dataframes

file_path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/notebook/'
with open(file_path+'label.pickle', 'rb')as f:
    encoder = pickle.load(f)

with open(file_path+'model.pickle', 'rb')as f:
    model = pickle.load(f)

results_mse = pd.read_csv(file_path+'df_result.csv', index_col='model')

results_prediction = pd.read_csv(file_path+'df_prediction.csv')

#plot feature importance
def feature_importance():

    features = ['start_month', 'Age', 'Ward_Name', 'From_WardId', 'Ward_Service_Type', 'Ward_Location', 'Gender',
                'TransferType']
    return dcc.Graph(
        figure={
            'data':[graph.Bar(y=features,
                              x=model[i].feature_importances_,
                              orientation='h',
                              name=i)
                    for i in ['rf','gbr','xgb']
                 ],
            'layout': graph.Layout(
                title={'text':'Feature Importance'},
                xaxis={'title':'Score'})
                # yaxis={'title':'Features'})
        },
    )


def prediction():
    df=results_prediction
    return dcc.Graph(
        id='prediction',
        className='graph',
        figure={
            'data':[
                graph.Scatter(
                    x=list(df.index),
                    y=df[i],
                    mode='markers',
                    opacity=.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.columns
            ],
            'layout': graph.Layout(
                title={'text':'Actual vs prediction'},
                xaxis={'title':'Episode Reference'},
                yaxis={'title':'Length of Stay'}
            )
        }
    )
