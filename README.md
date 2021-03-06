# Prediction of Length of Stay of Patients using Machine Learning
Source codes and documentation of my dissertation - Predict Length of Stay of Patients with 'Digital Ward'

This project aims to develop a mechanism for predicting length of stay of patients in the hospita, so as to facilitate operation management for Birmingham & Solihull Mental Health NHS Foundation Trust. The project is divided into three phases. Phase 1 and 2 involve developing prediction models, while Phase 3 involves developing a web application to visualise the data and deploy the model.
The project achieves this by adopting machine learning methodologies with data extracted from the inpatient database maintained by client as part of the ‘Digital Ward’ Project. The project begins with research into literatures and methodologies for regression analysis. The methodologies are incorporated into the model development process, which involves data pre-processing, features engineering, cross validation and testing. Finally, the model is deployed to a web application developed with Dash.

Below is a video of a summary of the project:
https://www.youtube.com/watch?v=HEN09-cMYfM&t=1s

## Instructions for running the web application:
1. Download (git clone) the deployed folder and all files inside it.**Do not modify the folder structure or delete any file.**

2. Before running the web application, make sure your system has met the **prerequisites listed below.**

3. Open terminal and run the following:
```
python3 [directory of the init.py folder]/init.py
```

4. The following message will appear, indicating that the web application has been loaded successfully and is now running on local server:
```
app:  <dash.dash.Dash object at 0x104962cf8>
Running on http://127.0.0.1:8050/
```

If the web browser does not start automatically, open and paste the address `http://127...` into web browser and open the web application.

## System Prerequisites
### Dash
This web application runs with the Dash library developed by Plotly. Dash's installation instruction and documentation can be accessed [here](https://dash.plot.ly/installation)

### Python version
It is recommended that **Python3 or above** shall be used to ensure compatibility.

### Python packages
The following Python packages need to be installed via the command `pip install *`, where * is replaced by the name of python package.

- numpy<br/>
- pandas<br/>
- seaborn<br/>
- matplotlib<br/>
- scikit-learn<br/>
- jenkspy<br/>
- xgboost<br/>
- scipy<br/>
- plotly<br/>






