import pandas as pd
import numpy as np
import random
import math
import pickle
import json
import os
import requests
import upload_to_s3
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

## FOR demo.ddl (DMM in Domino 4.6)
## USING CHURN-DMM-46 BUCKET

bucket = '<s3-bucket-name>'
model_id = '<dmm-model-id>'
dmm_api_key = os.environ.get('DMM_API_TOKEN')
daily_data_path = '/mnt/CustomerChurn46/DailyData/'

#Bring in data used to train pickled model that is loaded in later
df = pd.read_csv('/mnt/Test&TrainData/ChurnTrainingDataPP.csv')
df.drop('predictionProbability', axis =1,inplace = True)
print(df.shape)

#append data to istelf to double volume
df2 = df.append(df)

#Reset custid field so that there are no repeats
df2['custid'] = np.random.choice(range(df.custid.min(), df.custid.max()),df2.shape[0], replace=False)

##For each input feature adjust data and round/cast as necessary
#dropperc - 50%-150%
droppJitter = df2.dropperc.apply(lambda x : x*(random.randrange(50,150))/100)
#mins - 70%-130%
minsJitter = df2.mins.apply(lambda x : x*(random.randrange(70,130)/100)).round(0).astype(int)
#consecMonths - 80%-120%
consecMonthsJitter = df2.consecmonths.apply(lambda x : x*(random.randrange(80,120)/100)).round(0).astype(int)
#Income - 40%-160%
incomeJitter = df2.income.apply(lambda x : x*(random.randrange(40,160)/100)).round(1)
#age - 90%-110%
ageJitter = df2.age.apply(lambda x : x*(random.randrange(90,110)/100)).round(0).astype(int)


#Take all the new 'jittered' variables and write to a new df
#Keep original custid and churn_Y fields
df3 = pd.DataFrame({'custid': df2.custid,
       'dropperc': droppJitter, 
       'mins': minsJitter,
       'consecmonths': consecMonthsJitter,
       'income': incomeJitter,
       'age': ageJitter,
       'churn_Y': df2.churn_Y
                   })

#Understand correlations between new jittered data and original
#Should see larger data drift for lower correlations
concatset = pd.concat([df2,df3], axis =1)
concatset.columns=(list(df2.columns)+list('Jittered_'+df3.columns))
print('Correlations between original and altered vars -')
for i,k in enumerate(df2.columns):
    print('{}: {}'.format(k, concatset.corr()[k]['Jittered_'+k].round(4)))

#Load in trained model object    
loaded_model = pickle.load(open('/mnt/models/ChurnBinaryClassifier.pkl', 'rb'))

#Grab between 100 and 500 random rows from jittered data
df_inf = df3.sample(n = random.randint(100,500))
print(df_inf.shape[0], "records selected for sample")

#Save input features
X = df_inf.loc[:, 'dropperc':'age']

#Get model predictions for the sample of input features defined above
predictions = loaded_model.predict(X)
#Get model prediction probabilities for the sample of input features defined above
probas = loaded_model.predict_proba(X)

#write ground truth data out
churn_groundTruth = pd.DataFrame(df_inf[['custid','churn_Y']]).rename(columns = {'churn_Y': 'y_gt'})


#Create data set with predictions and pred probabilities
preds_df = pd.DataFrame(data=predictions, columns=['churn_Y'], index=churn_groundTruth.index)
preds_df['predictionProbability']= tuple(probas)

#adjust format of pred probabilities
preds_df['predictionProbability']=preds_df.predictionProbability.apply(lambda x : str(x).replace('(', '[').replace(')', ']'))


#join prediction and prob data with input features
churn_inputs_and_preds = df_inf.drop('churn_Y', axis =1)\
.join(preds_df, how = 'inner').drop_duplicates(subset = 'custid')

#Create 'protected class' feature to monitor for disparity in target distribution
churn_inputs_and_preds['Gender']=np.random.randint(0,3, churn_inputs_and_preds.shape[0])
mymap = {0:'M',1: 'F', 2: 'NB'}
churn_inputs_and_preds['Gender'] = churn_inputs_and_preds['Gender'].apply(lambda x: mymap.get(x))

#Add visualization of protected class for tracking purposes
ax = sns.barplot(x = ['F', 'M', 'NB'], y = churn_inputs_and_preds.groupby('Gender').mean()['churn_Y'])

ax.set_title('Churn By Gender')
ax.set_ylabel('Proportion')
ax.set_xlabel('Gender')
fig = ax.get_figure()
fig.set_size_inches(12,4)
plt.gcf().subplots_adjust(left=0.5)

fig.savefig('/mnt/Viz/protected_class_monitoring.png')

## Run two checks below to validate input and GT data is of same size and contains same custIDs

lenCheck = churn_inputs_and_preds.shape[0]==churn_groundTruth.shape[0]
print('Length Check Pass:', lenCheck)

idCheck= (churn_inputs_and_preds.custid.sort_values()==churn_groundTruth.custid.sort_values())\
.sum() ==churn_inputs_and_preds.shape[0]
print('ID Check Pass:', idCheck)

#Write input and pred data to DailyData folder
input_and_pred_path = str(daily_data_path+'inputs_and_preds_'+str(datetime.date.today())+'.csv')
churn_inputs_and_preds.to_csv(input_and_pred_path, index = False)

#Write GT data to DailyData folder
ground_truth_path = str(daily_data_path+'ground_truth_'+str(datetime.date.today())+'.csv')
churn_groundTruth.to_csv(ground_truth_path, index = False)

#Upload input&pred data and GT data to s3 bucket 
upload_to_s3.upload(input_and_pred_path, bucket)
upload_to_s3.upload(ground_truth_path, bucket)

#Define file names for payload construction below
inputs_file_name = str('inputs_and_preds_'+str(datetime.date.today())+'.csv')
ground_truth_file_name = str('ground_truth_'+str(datetime.date.today())+'.csv')

#Create inputs and prediction data payload
inputs_payload = """
{{
"datasetDetails": {{
        "name": "{0}",
        "datasetType": "file",
        "datasetConfig": {{
            "path": "{0}",
            "fileFormat": "csv"
        }},
        "datasourceName": "churn-dmm-46",
        "datasourceType": "s3"
    }}
}}
""".format(inputs_file_name)

#Define api endpoint
inputs_url = "https://demo.dominodatalab.com/model-monitor/v2/api/model/{}/register-dataset/prediction".format(model_id)

#Set up call headers
headers = {
           'X-DMM-API-KEY': dmm_api_key,
           'Content-Type': 'application/json'
          }
#Make API call
inputs_response = requests.request("PUT", inputs_url, headers=headers, data = inputs_payload)

#Print response (successfull call will have no response)
print(inputs_response.text.encode('utf8'))
 
#create GT payload    
ground_truth_payload = """
{{

"datasetDetails": {{
        "name": "{0}",
        "datasetType": "file",
        "datasetConfig": {{
            "path": "{0}",
            "fileFormat": "csv"
        }},
        "datasourceName": "churn-dmm-46",
        "datasourceType": "s3"
    }}
}}
""".format(ground_truth_file_name)

#Define api endpoint
ground_truth_url = "https://demo.dominodatalab.com/model-monitor/v2/api/model/{}/register-dataset/ground_truth".format(model_id)

#Make api call
ground_truth_response = requests.request("PUT", ground_truth_url, headers=headers, data = ground_truth_payload)

#Print response
print(ground_truth_response.text.encode('utf8'))
