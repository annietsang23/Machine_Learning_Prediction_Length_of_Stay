#Part of the codes are adopted from Dash User Guide and Documentation published by Plotly
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input,Output,State
import pandas as pd
import numpy as np
import matplotlib as plotlib
import plotly.graph_objs as graph
import io
import base64
import datetime
import time
import json
import create_df as cf
import mi
import sys
import pickle
import urllib.parse

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
print('app: ',app)

def timer(f):
    def wrapper(*args,**kwargs):
        start=time.time()
        result=f(*args,**kwargs)
        end=time.time()
        print(f'{f.__name__} - Processing time: {end-start}s')
        return result
    return wrapper

@timer
def generate_scatter_graph(df,x='WardName'):
    return dcc.Graph(
        id='scatter_all',
        className='graph',
        figure={
            'data': [
                graph.Scatter(
                    x=df[df.WardName==i][x],
                    y=df[df.WardName==i]['los_days'],

                    mode='markers',
                    opacity=0.7,
                    # marker={
                    #     'size': 5,
                    #     'line': {'width': 0.5, 'color': 'white'}
                    # },
                    name=str(i)
                ) for i in df.WardName.unique()
            ],
            'layout': graph.Layout(
                title={'text':f'{x} distribution'},
                xaxis={'title': f'{x}'},
                yaxis={'title': 'length of stay (days)'},
            )
        })


def create_list_of_objects(col):
    l= []
    for i in col:
        k = {}
        k['label'] = i
        k['value'] = i
        l.append(k)
    return l

def df_to_dict(df,type):
    data=[]
    for i in df.columns:
        v={}
        v['x']=list(df.index)
        v['y']=list(df[i])
        v['type']=type
        v['name']=i
        data.append(v)
    return data

#export figure
def export_figure(fig,format='png'):
    '''
    export figure to file
    :param fig: figure object
    :param format: string. Acceptable values include: png,jpeg,webp,svg,pdf
    :return: None
    '''
    filename='images/'+fig.id+'.'+format
    fig.write_image(filename)


# #craet list of all attributes in the dataframe
# attributes_list=create_list_of_objects(df.columns)

#create list of ward no.
# ward_list=create_list_of_objects(set(df.Ward))


markdown_text='''
This is a dashboard and graph created by using the data from inpatient database.
'''
#layout of the webpage
app.layout = html.Div(children=[
    html.H1(children='Inpatient Database Dashboard'),
    #markdown text
    dcc.Markdown(markdown_text),

    # upload file
    html.Div([
        dcc.Upload(
        id='upload-file',
        children=html.Div([
                    """Upload files (filenames have to match the following): 
                    \nward_type.csv 
                    \nPatientsDemographics.csv
                    \nPatientActivities.csv\n""",

            html.A('Select files')
        ]),
        # Allow multiple files to be uploaded
        multiple=True
    ),
    #for uploading file

    html.Div(id='file-uploaded')]),
    #store the dataframe as json object

    html.Div(id='store-df',style={'display':'none'}),
    html.Div(id='store-df_pa',style={'display':'none'}),
    html.Div(id='store-df_wt',style={'display':'none'}),
    html.Div(id='store-df_pd',style={'display':'none'}),

    #generate the tabs
    html.Div([
        dcc.Tabs(id='tabs',value='table',children=[
            dcc.Tab(label='Summary',value='summary'),
            dcc.Tab(label='Per Ward',value='ward'),
            dcc.Tab(label='Predict length of stay',value='ml'),
        ]),
        html.Div(id='tabs-content')
    ]),

])

app.config.suppress_callback_exceptions = True



def upload_file(f_name,f_content,last_modified):
    c_t,c_s=f_content.split(',')
    # print('c_s: ',c_s)
    try:
        if 'xls' in f_name:
            df=pd.read_excel(io.BytesIO(base64.b64decode(c_s)))
        elif 'csv' in f_name:
            df = pd.read_csv(io.StringIO(base64.b64decode(c_s).decode('utf-8')))
        else:
            return html.Div(['Only xls or csv file format allowed.']),['N']

        # print('df: ',df.head())

        # df=df.to_json(date_format='iso',orient='split')

        # print('type df: ',type(df))
        return html.Div([
            html.Plaintext(f'file name: {f_name}\t last modified: {datetime.datetime.fromtimestamp(last_modified)}'),
        ]),df
    except OSError:
        return html.Div(['File not found / cannot be uploaded.']),['N']

@timer
#process uploaded files
@app.callback([Output('file-uploaded','children'),
              Output('store-df_pa','children'),
              Output('store-df_wt','children'),
              Output('store-df_pd','children'),
              Output('store-df', 'children')],
              [Input('upload-file','contents')],
              [State('upload-file','filename'),
               State('upload-file','last_modified')])

def get_file(contents,filename,last_modified):

    if not contents is None:
        # print('contents: ',contents)
        # print('filename:', filename)
        # print('last_modified', last_modified)

        for i in range(len(filename)):
            if filename[i]=='PatientActivities.csv':
                file_pa=i
            elif filename[i]=='ward_type.csv':
                file_wt=i
            elif filename[i]=='PatientsDemographics.csv':
                file_pd=i
            else:
                return html.Div(['Files accepted are: PatientActivities.csv, ward_type.csv, PatientsDemographics.csv']),['N'],['N'],['N']

        c_l=[None]*len(filename)
        df_l=[None]*len(filename)

        # print('len',range(len(contents)))
        for i in range(len(filename)):
            c_l[i],df_l[i]=upload_file(filename[i],contents[i],last_modified[i])

        # return c[0][0],c[0][1]
        # print(c_l)

        #create and combine df
        df_pa=cf.patient_activities_df(df_l[file_pa])
        df=cf.combine_df(df_pa,df_l[file_wt],df_l[file_pd])

        df_json=[i.to_json(date_format='iso',orient='split') for i in (df_pa,df_l[file_wt],df_l[file_pd],df)]

        return c_l,df_json[0],df_json[1],df_json[2],df_json[3]

    else:
        return html.Div(['File not yet uploaded']),['N'],['N'],['N'],['N']


#Change view based on tab selection
@app.callback(Output('tabs-content','children'),
              [Input('tabs','value'),
               Input('store-df','children'),
               Input('store-df_pa','children')])
def render_tab_content(tab,df_json,df_json_pa):
    if not (((df_json ==['N']) or (df_json is None)) or ((df_json_pa ==['N']) or (df_json_pa is None))):
        df=pd.read_json(df_json,orient='split')
        df_pa = pd.read_json(df_json_pa,orient='split')
        df_pa = cf.patient_activities_df(df_pa, iso=True)

        if tab=='summary':
            #plot admission,transfer and discharge activities over the year
            if ((not df_pa.empty) and (not df.empty)) and (isinstance(df_pa, pd.core.frame.DataFrame)
                    and isinstance(df,pd.core.frame.DataFrame)):
                return html.Div(children=[

                    # data download link
                    html.A(
                        'Download data',
                        id='download-summary',
                        download="summary.csv",
                        href="",
                        target="_blank"
                    ),

                    cf.plot_activities_over_time(df_pa, None),

                    html.Div([
                       # checkboxes for selecting attributes
                       html.Label('Select attribute:'),
                       dcc.RadioItems(
                           id='attribute_selected',
                           options=create_list_of_objects(df.columns),
                           style={'columnCount': 2},
                           value='WardName'
                       )],
                     ),
                    html.H3(children='Plot attributes against length of stay'),
                    dcc.RadioItems(
                        id='graph-selected',
                        options=create_list_of_objects(['boxplot','scatterplot']),
                        style={'columnCount':2},
                        value='boxplot'
                    ),
                    html.Div(id='summary-graph'),
                    # dcc.Graph(id='boxplot_all'),
                    # generate_scatter_graph(df)
                    # generate_table(df)
                    ]),
        elif tab =='ward':
            #generate contents for ward
            return html.Div([
                    html.Label('Select ward:'),

                    dcc.Checklist(
                        id='ward_selected',
                        options=create_list_of_objects(set(list(df.WardName))),
                        style={'columnCount': 3}
                        ),
                    #data download link
                    html.A(
                        'Download Ward data',
                        id='download-ward',
                        download="",
                        href="",
                        target="_blank"
                    ),

                    #show the wards selected
                    html.H1(id='ward-names'),
                    html.Div(id='table-style-outer',children=[

                        html.Div([html.P('Average Turnaround (days): '),
                                  html.H4(id='text_avg_los')],
                                  className='table-style'),

                        html.Div([html.P('Average Age: '),
                                  html.H4(id='text_avg_age')],
                                 className='table-style'),

                        html.Div([html.P('Average number of patients admitted per month: '),
                                  html.H4(id='text_no_patients')],
                                 className='table-style'),

                    ]),

                    html.Div([
                        dcc.Graph(id='hist_los', className='table-style'),
                        dcc.Graph(id='hist_age', className='table-style'),
                        dcc.Graph(id='hist_gender', className='table-style')]),

                    html.Div(id='plot-activities'),

                    dcc.RadioItems(
                        id='attribute_selected_ward',
                        options=create_list_of_objects(df.columns),
                        style={'columnCount': 2},
                        value='Age'
                        ),


                    #boxplot
                    dcc.Graph(id='boxplot'),
                    html.Div(id='table_all')
                    # generate_table(df)



                ]),

        elif tab=='ml':
            #generate machine learning training result
            return html.Div([
                html.H4('Prediction of length of stay'),
                 html.Div([
                     html.H5('Enter values below:'),
                     html.Label('Transfered from which ward?'),
                     dcc.Dropdown(
                         id='mi-fromward',
                         options=create_list_of_objects(['None']+list(set(df.WardName)))
                     ),
                     html.Label('Select transfer type: '),
                     dcc.Dropdown(
                         id='mi-transfertype',
                         options=create_list_of_objects(list(set(df.TransferType)))
                     ),
                     html.Label('Current ward:'),
                     dcc.Dropdown(
                         id='mi-ward',
                         options=create_list_of_objects(list(set(df.WardName)))
                     ),
                     html.Label('Current month'),
                     dcc.Dropdown(
                         id='mi-startmonth',
                         options=create_list_of_objects(list(range(1,13)))
                     ),
                     html.Label('Age of Patient: '),
                     dcc.Input(
                         id='mi-age',
                         type="number"
                     ),
                     html.Label('Gender of Patient: '),
                     dcc.RadioItems(
                         id='mi-gender',
                         options=create_list_of_objects(['M','F'])
                     ),
                ]),
                 html.Button('Submit',id='submit'),
                 html.Label(id='hint'),

                 html.H3('Predicted stay (no. of days)'),
                 html.Div(id='mi-prediction', children=[

                     html.Div([html.P('Random Forest'),
                               html.H4(id='mi-rf')],
                              className='table-style'),

                     html.Div([html.P('Gradient Boosting'),
                               html.H4(id='mi-gb')],
                              className='table-style'),

                     html.Div([html.P('XGBoost'),
                               html.H4(id='mi-xgb')],
                              className='table-style'),

                 ]),
                html.H3('Training and Testing Result'),
                 #feature importance
                mi.feature_importance(),
                #traninig result
                 testing_result(),
                 mi.prediction(),

            ])
        else:
            return html.Div(['Click tabs to see data'])
    else:
        return html.Div(['Please upload file first'])

counter=1

@app.callback(
    [Output('mi-rf','children'),
     Output('mi-gb','children'),
     Output('mi-xgb','children'),
     Output('hint','children')],
    [Input('submit','n_clicks'),
     Input('mi-fromward','value'),
     Input('mi-transfertype','value'),
     Input('mi-ward','value'),
     Input('mi-startmonth','value'),
     Input('mi-age','value'),
     Input('mi-gender','value'),
     Input('store-df_wt','children')])
def make_prediction(n_clicks,from_ward,transfer_type,ward,start_month,age,gender,df_wt_json):
    global counter
    print('clicks: ',counter,n_clicks)
    print(from_ward,transfer_type,ward,start_month,age,gender)
    if counter==n_clicks:
        counter+=1
        for i in (from_ward,transfer_type,ward,start_month,age,gender):
            if i is None:
                return "","","","Missing values. Attribute cannot be empty"
        if age<0:
            return "","","","Age cannot be negative"

        # Get ward location and service type data from ward_type.csv
        if (df_wt_json is None):
            return html.P('ward_type.csv file not yet uploaded. Please upload file.')
        df_wt=pd.read_json(df_wt_json,orient='split')
        df_wt.columns = [col.replace(' ', '') for col in df_wt.columns]
        ward_location=df_wt[df_wt.WardName==ward].WardLocation.values[0]
        ward_service_type=df_wt[df_wt.WardName==ward].WardServiceType.values[0]

        print(ward_location,ward_service_type)

        gender_e=mi.encoder['label']['Gender'].get(gender,0)
        ward_service_type_e=mi.encoder['label']['Ward Service Type'].get(ward_service_type,0)
        ward_location_e=mi.encoder['label']['Ward Location'].get(ward_location,0)
        from_ward_e=mi.encoder['label']['Ward'].get(from_ward,0)
        ward_e=mi.encoder['label']['Ward'].get(ward,0)
        age_e=int(np.digitize([42],mi.encoder['age'])[0])+1

        attributes={'From_WardId':[from_ward_e],
           'TransferType':[transfer_type],
           'start_month':[start_month],
           'Ward_Name': [ward_e],
           'Ward_Service_Type':[ward_service_type_e],
           'Ward_Location':[ward_location_e],
           'Age':[age_e],
           'Gender':[gender_e]}
        X=pd.DataFrame.from_dict(attributes)

        prediction_rf=int(np.round(mi.model['rf'].predict(X),2))
        prediction_gbr = int(np.round(mi.model['gbr'].predict(X),2))
        prediction_xgb = int(round(np.round(mi.model['xgb'].predict(X),2)[0],2))

        return prediction_rf,prediction_gbr,prediction_xgb,"Attributes are valid"
    else:
        return "","","",""
def testing_result():

    return dcc.Graph(
        id='result',
        figure={
            'data': df_to_dict(mi.results_mse,'bar'),
            'layout':{
                'title': 'Result Comparison - Mean Squared Error'
            }
        }
    )

#Tab Summary: update download link for all data
@app.callback(
     Output('download-summary', 'href'),
    [Input('store-df','children')])
def update_download_summary(df_json):
    if not df_json=="":
        df = pd.read_json(df_json, orient='split')
        download_href = create_download_link(df)
        return download_href

#Tab Summary: plot activities graph based on ward selection and update download link for data
@app.callback(
    Output('plot-activities','children'),
    [Input('store-df_pa','children'),
     Input('ward_selected','value')])
def plot_activities(df_pa,ward_selected):
    df_pa=pd.read_json(df_pa,orient='split')
    df_pa = cf.patient_activities_df(df_pa, iso=True)
    return cf.plot_activities_over_time(df_pa,ward=ward_selected)

#Tab Summary_los: plot graphs of attributes against length of stay for all wards
@app.callback(
    dash.dependencies.Output('summary-graph','children'),
    [dash.dependencies.Input('graph-selected','value'),
     dash.dependencies.Input('store-df','children'),
     dash.dependencies.Input('attribute_selected','value')])
def show_summary_graph(graph_selected,df_json,attribute_selected):
    df = pd.read_json(df_json, orient='split')
    if graph_selected=='boxplot':
        return plot_boxplot_all(df,attribute_selected)
    elif graph_selected=='scatterplot':
        return generate_scatter_graph(df,attribute_selected)

def plot_boxplot_all(df,col='WardName'):

    df_a = pd.DataFrame()
    df_a['EpisodeReference'] = df.EpisodeReference
    df_a['WardName']=df.WardName
    for i in df[col].unique():

        if df_a.shape == (0, 0):
            df_a[i] = df[df[col] == i].los_days
        else:
            df_x = pd.DataFrame()
            df_x['EpisodeReference'] = df.EpisodeReference
            df_x[i] = df[df[col] == i].los_days
            df_a = pd.merge(df_a, df_x, on='EpisodeReference', how='outer')
            df_a.drop_duplicates(inplace=True)
    df_n = df_a.drop(['EpisodeReference','WardName'], axis=1).columns

    return dcc.Graph(
        id='boxplot_all',
        className='graph',
        figure={
            'data':[graph.Box(
                y=df_a[j].values,
                # name=str(df_a[df_a[j].notnull()].Ward)
                name=str(j)
                ) for j in df_n
            ],
            'layout': graph.Layout(
                title=f'{col} distribution',
                xaxis={'title': f'{col}', 'showgrid': False},
                yaxis={'title': 'Length of stay (Days)', 'showgrid': False}
            )
        })

#Tab: Per Ward. Display summary info adn plot graphs

#Display Ward Names, no. of patients, avg length of stay and avg age as Text, as well as change value of download link
@app.callback(
    [Output('ward-names','children'),
     Output('text_avg_los','children'),
     Output('text_avg_age','children'),
     Output('text_no_patients','children'),
     Output('download-ward','href'),
     Output('download-ward','download')],
    [Input('store-df','children'),
     Input('ward_selected', 'value')])
def show_summary_info_link(df_json,ward_selected):
    df = pd.read_json(df_json, orient='split')
    if (ward_selected!=[]) and (ward_selected is not None):
        ward_names=', '.join(ward_selected)
        df_filtered=df[df.WardName.isin(ward_selected)]
        avg_los=int(df_filtered.los_days.mean())
        avg_age=int(df_filtered.Age.mean())
        no_patients=int(df_filtered.start_month.value_counts().mean())

        text_los=f'Average turnaround: {avg_los} days'
        text_age=f'Average Age: {avg_age}'
        text_no_patients=f'Average no. of patients intake per month: {no_patients}'

        # create download link href and filename
        download_href=create_download_link(df_filtered)
        download_fn='_'.join(ward_selected)+".csv"

        # return text_los,text_age,text_no_patients,download link href and filename
        return ward_names,avg_los,avg_age,no_patients,download_href,download_fn
    else:
        download_href = create_download_link(df)
        download_fn = "all_wards.csv"
        return '','','','',download_href,download_fn

def plot_hist_ward(df,cols,col='los_days',bin_x=15):
    return {
            'data': [graph.Histogram(
                    x=df[df.WardName==i][col],
                    opacity=0.7,
                    name=str(i),
                    marker={"line": {"color": "#25232C", "width": 0.2}},
                    xbins={"size":bin_x},
                    customdata=[i]
                ) for i in cols
            ],
            'layout': graph.Layout(
                title=f'{col} distribution',
                xaxis={'title': f'{col}','showgrid':False},
                yaxis={'title':'Count','showgrid':False}
            )
        }

def plot_hist_all(df,col='los_days',bin_x=15):
    return {
            'data': [graph.Histogram(
                    x=df[col],
                    opacity=0.7,
                    name=col,
                    marker={"line": {"color": "#25232C", "width": 0.2}},
                    xbins={"size":bin_x},
                    customdata=df[col]
                )
            ],
            'layout': graph.Layout(
                title=f'{col} distribution',
                xaxis={'title': f'{col}','showgrid':False},
                yaxis={'title':'Count','showgrid':False}
            )
        }

def plot_boxplot_ward(df,selected,col):
    df_a = pd.DataFrame()
    df_ward = df[df.WardName.isin(selected)]
    df_a['EpisodeReference'] = df_ward.EpisodeReference
    for i in df_ward[col].unique():

        if df_a.shape == (0, 0):
            df_a[i] = df_ward[df_ward[col] == i].los_days
        else:
            df_x = pd.DataFrame()
            df_x['EpisodeReference'] = df_ward.EpisodeReference
            df_x[i] = df_ward[df_ward[col] == i].los_days
            df_a = pd.merge(df_a, df_x, on='EpisodeReference', how='outer')
            df_a.drop_duplicates(inplace=True)
    df_n=df_a.drop('EpisodeReference',axis=1).columns


    return  {
            'data':[graph.Box(
                y=df_a[j].values,
                name=str(j),
                customdata=df_a[df_a[j].notnull()].EpisodeReference
                ) for j in df_n
            ],
            'layout': graph.Layout(
                title=f'{col} distribution',
                xaxis={'title': f'{col}', 'showgrid': False},
                yaxis={'title': 'Length of stay (Days)', 'showgrid': False}
            )
        }

#Update Table based on ward selection and click on boxplot
@app.callback(
    dash.dependencies.Output('table_all','children'),
    [dash.dependencies.Input('store-df','children'),
     dash.dependencies.Input('boxplot','clickData'),
     dash.dependencies.Input('ward_selected', 'value')])
def update_table(df_json,clickData,ward_selected):
    # print(clickData)
    # print(ward_selected)
    df=pd.read_json(df_json,orient='split')
    df_filtered =df.copy()

    if (ward_selected!=[]) and (ward_selected is not None):
        df_filtered=df[df.WardName.isin(ward_selected)]

    if not clickData is None:
        idx = clickData['points'][0]['customdata']
        df_filtered = df[(df.EpisodeReference ==idx)]

    data = df_filtered.to_dict('records')

    return generate_table(df_filtered,data)

def generate_table(df,data,max_rows=20):
    return dash_table.DataTable(
        # id='table_all',
        columns=[{'name':i,'id':i} for i in df.columns],
        data=data,
        style_cell={'textAlign':'left'},
        style_cell_conditional=[
            {
                'if':{'column_id':'EpisodeReference'},
                'textAlign':'left'
            }
        ],
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        style_table={
            'height': '300px',
            'overflowY':'scroll'
        },
        sort_action='native',
        sort_mode='multi',
        filter_action='native',
    )

#Tab: Per Ward. Update histogram based on ward selection
@app.callback(
    dash.dependencies.Output('hist_los','figure'),
    [dash.dependencies.Input('store-df','children'),
     dash.dependencies.Input('ward_selected','value')])
def update_hist_ward(df_json,ward_selected):
    if (ward_selected != []) and (ward_selected is not None):
        df=pd.read_json(df_json, orient='split')
        return plot_hist_ward(df,cols=ward_selected,col='los_days')
    else:
        return {
            'data': [],
            'layout': graph.Layout(
                title=f'No ward selected',
            )
        }

@app.callback(
    dash.dependencies.Output('hist_age','figure'),
    [dash.dependencies.Input('store-df','children'),
    dash.dependencies.Input('ward_selected','value')])
def update_hist_ward(df_json,ward_selected):
    if (ward_selected != []) and (ward_selected is not None):
        df = pd.read_json(df_json, orient='split')
        return plot_hist_ward(df,cols=ward_selected,col='Age')
    else:
        return {
            'data':[],
            'layout': graph.Layout(
                title=f'No ward selected',
            )
        }

@app.callback(
    dash.dependencies.Output('hist_gender','figure'),
    [dash.dependencies.Input('store-df','children'),
    dash.dependencies.Input('ward_selected','value')])
def update_hist_ward(df_json,ward_selected):
    if (ward_selected != []) and (ward_selected is not None):
        df = pd.read_json(df_json, orient='split')
        return plot_hist_ward(df,cols=ward_selected,col='Gender')
    else:
        return {
            'data': [],
            'layout': graph.Layout(
                title=f'No ward selected',
            )
        }

#Ward: Update boxplot based on Ward and Attribute selection
@app.callback(
    dash.dependencies.Output('boxplot','figure'),
    [dash.dependencies.Input('store-df','children'),
    dash.dependencies.Input('ward_selected','value'),
     dash.dependencies.Input('attribute_selected_ward','value')])
def update_boxplot(df_json,ward_selected,attribute_selected):
    if (ward_selected != []) and (ward_selected is not None):
        df = pd.read_json(df_json, orient='split')
        #default ward and attribute selection
        w=[df.WardName.unique()[0]]
        a='Age'

        w=ward_selected
        if not attribute_selected is None:
            a=attribute_selected

        return plot_boxplot_ward(df,w,a)
    else:
        return {
            'data':[],
            'layout': graph.Layout(
                title=f'No ward selected',
            )
        }

def create_download_link(df):
    csv_str=df.to_csv(index=False,encoding='utf-8')
    print(type(csv_str))
    csv_str = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_str)
    print(csv_str)
    return csv_str

if __name__ == "__main__":
    #debug=True enables the browser to be automatically refreshed when changes are made to the code.
    app.run_server(debug=True)