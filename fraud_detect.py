import csv
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

#funtion to return IP data
def readIPData(file):
    #read in IP Addresses
    data = pd.read_csv(file)
    #sort by lower bound
    data_sorted = data.sort_values(['lower_bound_ip_address','upper_bound_ip_address'], ascending=['True','True'])
    #return IP data
    return(data)

#funtion to return fraud data
def readFraudData(file):  
    #read in IP Addresses
    data = pd.read_csv(file)
    #sort by user id
    data_sorted = data.sort_values(['user_id'], ascending=['True'])
    #return fraud data
    return(data_sorted)

#funtion to assign country by IP address
def assignCountry(frauddata, ipdata): 
    #initialize column
    data = pd.DataFrame(index=frauddata.index, columns=['country'])
    #loop through ipaddress
    for index, row in frauddata.iterrows():
        #find index within bounds
        ip = row['ip_address']
        lowerb = ipdata.loc[ipdata['lower_bound_ip_address'] <= ip]
        upperb = lowerb.loc[ipdata['upper_bound_ip_address'] >= ip]
        #check if only one country fits description
        if len(upperb['country'])==1:
            data.loc[index,'country'] = upperb['country'].iloc[0]
        else:
            data.loc[index,'country'] = "Unknown"
    #return country data frame
    return(data)

#function for cleaning data (mixed)
def getData(data): 
    #initialize column
    data_mixed = pd.DataFrame(index=data.index)
    #get time from sign up to purchase
    time = pd.to_datetime(data['purchase_time'])-pd.to_datetime(data['signup_time'])
    time = time/np.timedelta64(1, 's')
    #get device_id counts
    data['device_count'] = data.groupby('device_id')['device_id'].transform('count')
    #get ip address counts
    data['ip_count'] = data.groupby('ip_address')['ip_address'].transform('count')
    #convert categorical data to numeric categories
    source = pd.factorize(data['source'])[0]
    browser = pd.factorize(data['browser'])[0]
    sex = pd.factorize(data['sex'])[0]
    country = pd.factorize(data['country'])[0]
    #create dataframe
    data_mixed = data_mixed.assign(time=time, purchase_value=data['purchase_value'], 
                                 device=data['device_count'], source=source, browser=browser,  
                                 age=data['age'], ipadd=data['ip_count'])#, country=country)
    #return data
    return(data_mixed)


#read in data
ipdata = readIPData("Fraud/IpAddress_to_Country.csv")
frauddata = readFraudData("Fraud/Fraud_Data.csv")

#check for duplicates entries
print(frauddata.duplicated().any()) #all users unique? Yes

#include country information 
country = assignCountry(frauddata, ipdata)
data = frauddata.assign(country=country)
print(data.head())

#explore data
print(data['device_id'].duplicated().any()) #all devices unique? No
print(data['ip_address'].duplicated().any()) #all ip addresses unique? No

#process time, device, and ip_address data
#convert categorical data strings to numeric categories
dataset = getData(data)

#split into training and test groups
X_train, X_test, y_train, y_test = train_test_split(dataset, data['class'], test_size = 0.33)

#build model 
#randomforest (appropriate for mix of categorical and numeric data, non-parametric, previously shown to be successful on binary classification)
#would try several other models if time allowed (ie Logistic Regression, Multilayer Perceptron, etc.)
#would optimize features if time allowed (ie number of features, which feautres, etc)
#would optimize model parameters if time allowed (ie number of trees, class thresholds, etc)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)
print(auc)


#Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='red',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Fraud Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


#2) In the case of fraud, false positives are likely preferred. Flagging
#non-fraud cases just requires a human verification, but failing to flag
#fraud results in loss of money/investigation/etc.


#3) The model looks at features of the account including time from signing
#up to purchase, sex, age, etc. from previous cases of fraud and non-fraud.
#The model then predicts whether an account is behaving more similarly to 
#cases of fraud or non-fraud and flags them.


#4) You can use the product to alert the company if a person's account 
#is flagged as potentially fradulent. This could automatically freeze the
#account until a human reviews the data


#If fraud probability<X, normal experience
#If X<=fraud probability<Z, extra verification step
#IF fraud probability>Z, human review

