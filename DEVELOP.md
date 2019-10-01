# Development
This page documents how to further develop the machine learning prediction models and web application. 
For project overview and user instruction, please see the [README](./README.md) for more information.

## Setup

The **system prerequisites** listed in [README](./README.md) must be met.

The source codes for developing the prediction model are written in **jupyter notebook**. It can be downloaded [here](https://jupyter.org/).

Below lists the version of all the packages used in development.
```
python: 3.6.9
jupyter notebook: 5.7.4
numpy: 1.15.4
pandas: 0.23.4
seaborn: 0.9.0
matplotlib: 3.0.2
scikit-learn: 0.20.0
jenkspy: 0.1.5
xgboost: 0.90
scipy: 1.2.0
plotly: 2.5.1
dash: 0.21.0
dash-core-components: 0.22.1
dash-html-components: 0.10.0
dash-renderer: 0.12.1
```
## Edit the Models
The following notebooks, saved under[notebook](‘./deployed/notebook’), are used to develop the model:
- **Patient Activities**:
▪ It imports the data from csv files, calculate the length of stay variable and concatenate them into one single pandas dataframe.
▪ The resulted dataframe is saved as ‘df_los_p_o.csv’ in the same directory.
- **Feature Engineering**:
▪ It imports the ‘df_los_p_o.csv’ file from the same directory, perform feature analysis, label encoding etc.
▪ It then pickles and saves the encoder used in the same directory for later use in the web application.
▪ When the feature set has been built, it imports the functions from Build Model notebook and run cross validation. The results are visualised.
▪ The best models are pickled and saved in the same directory for later use in the web application.

- **Build Model**
▪ It stores models to be built in different functions, which are imported by Feature Engineering notebook.
- Visualisation
▪ It plots the feature importance from the optimal models built in Feature Engineering notebook.

To start jupyter notebook, type the following in command line:
`$ jupyter notebook`

The notebook shall be run in the order of PatientActivities -> Feature Engineering.

Csv files and pickled files are created and saved automatically as the cells are run. Please do not delete any files created.


## Edit the web application
The web application is created using Dash Plotly, a Python framework for developing web apps for data analysis. Please refer to [Dash's documentation](https://dash.plot.ly/getting-started) for further details.

### Folder Structure

The source codes of the web application are saved under [IXN_dash](./deployed/IXN_dash). Below is the structure of the folder:
```
|-- IXN_dash
|   |--create_df.py
|   |--init.py
|   |--mi.py
|   |--assets
|   |  |--LOS.css
```
**Init.py** is used to start and run the application on the local server (in line 777). Debug is set to True to enable debugging during development.
```
if __name__ == "__main__":
    #debug=True enables the browser to be automatically refreshed when changes are made to the code.
    app.run_server(debug=True)
```
The **init.py** file imports **create_df.py** and **mi.py**. The **create_df.py** file is used to process/create the dataframes from source files uploaded by user. The **mi.py file** is used to import and unpickle encoder.pickle and model.pickle built from model development, and run prediction.

### CSS and Javascript

To run css and javascript from separated files, the files need to be stored under ‘**assets**’. A css file named **LOS.css** has already been created to specify the color and layout of the web application.

### Callbacks
Interactivity in Dash is achieved through callbacks ([See Documentation](https://dash.plot.ly/getting-started-part-2)). For example, a callback decorator can be added to a function, which takes the user-selected checkbox values i.e. Ward name(s) etc. as input variables and return the filtered plots as output. This enables the application to accept inputs from user and dynamically change the views.
```
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
 ```

When the application is initialised, all the callbacks are initialised. After that, callback will be fired if there’s any change in values to input variables.



