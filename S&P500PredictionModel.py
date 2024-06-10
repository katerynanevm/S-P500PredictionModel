#!/usr/bin/env python
# coding: utf-8

# In[24]:


get_ipython().system('pip install yfinance')
import yfinance as yf


# In[25]:


sp500 = yf.Ticker("^GSPC")


# In[26]:


sp500 = sp500.history(period="max")


# In[27]:


sp500


# In[28]:


sp500.index


# In[29]:


#Show up the plot of the closing price for the whole period 
sp500.plot.line(y="Close", use_index=True)


# In[30]:


#Deleating columns we dont need 
del sp500["Dividends"]
del sp500["Stock Splits"]


# In[35]:


#Creates the column Tomorrow, where will be shown tomorrows closed price
# Дати ідуть від еайстаршого до найближчого, тому шоь вщяти данні на завтра, 
#ми ьеремо данні із наступної колонки 
sp500["Tomorrow"] = sp500["Close"].shift(-1)


# In[32]:


sp500


# In[33]:


#Craetes column called Target, where will be shown 
#whether the tomorrows price is higher than todays price
#if yes, than it will be shown as an intiger 1
sp500["Target"] = (sp500["Tomorrow"] > sp500["Close"]).astype(int)


# In[34]:


sp500


# In[36]:


#.loc[] is a label-based indexer for selecting rows and columns by label(s) or a boolean array.
#The colon (:) indicates that you want to include all columns.
#The .copy() method creates a copy of the filtered DataFrame.
sp500 = sp500.loc["1990-01-01":].copy()


# In[37]:


sp500


# In[40]:


from sklearn.ensemble import RandomForestClassifier

#n_estimators=100: This sets the number of trees in the forest to 100.
#min_samples_split=100: This specifies the minimum number of samples required 
#to split an internal node. A higher value can prevent the model from overfitting.
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = sp500.iloc[:-100]  #selects all rows of the sp500 DataFrame except the last 100 rows for the training set.
test = sp500.iloc[-100:]   #selects all rows of the sp500 DataFrame except the last 100 rows for the training set.

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])


# In[55]:


from sklearn.metrics import precision_score

preds = model.predict(test[predictors]) #test[predictors]:This selects the columns specified in the predictors list from the test DataFrame.
                                        


# In[83]:


preds


# In[56]:


import pandas as pd 
preds = pd.Series(preds, index=test.index)  #converts the array of predictions into a pandas Series.
                                            #index=test.index:This sets the index of the new Series to match the index of the test DataFrame.
                                            #test.index provides the index labels from the test DataFrame.


# In[59]:


preds


# In[67]:


from sklearn.metrics import accuracy_score, precision_score

# Evaluate the model's performance
accuracy = accuracy_score(test["Target"], preds)
print(f"Accuracy: {accuracy:.2f}")


# In[ ]:


#Accuracy
#Definition: Accuracy is the proportion of correctly predicted instances (both true positives and true negatives) out of the total number of instances.
#Formula: 
#Accuracy
#  = Number of Correct Predictions/ Total Number of Predictions
#Interpretation: Accuracy provides an overall measure of how often the model makes correct predictions across all classes.


# In[68]:


# Print the distribution of predictions
print(preds.value_counts())


# In[75]:


precision = precision_score(test["Target"], preds, zero_division=0)


# In[76]:


precision


# In[77]:


# Calculate and print the precision score
precision = precision_score(test["Target"], preds, zero_division=0)
print(f"Precision: {precision:.2f}")


# In[80]:


#Precision
#Definition: Precision is the proportion of true positive predictions out of the total predicted positives. It focuses on the accuracy of positive predictions.
#Formula: Precision = True Positives/ (True Positives + False Positives)
#Interpretation: Precision tells us how precise or accurate the positive predictions made by the model are. It measures the model's ability to avoid false positives.


# In[81]:


combined = pd.concat([test["Target"], preds], axis=1)


# In[82]:


combined.plot()


# In[84]:


# The code combined = pd.concat([test["Target"], preds], axis=1) concatenates two pandas Series (test["Target"] and preds) into a single DataFrame along the columns (axis=1). Here’s a breakdown of what each part of the code does:

# Breakdown of the Code

# pd.concat():
# This function concatenates pandas objects along a particular axis.
# It can concatenate Series, DataFrame, or a list of these objects.

# [test["Target"], preds]:
# This is a list containing the two pandas Series that you want to concatenate.
# test["Target"] represents the actual target values from the test set.
# preds represents the predicted values obtained from the model.
# axis=1:

# This parameter specifies the axis along which the concatenation should be performed.
# axis=1 means the concatenation is done along the columns, meaning that the two Series will be aligned side by side in the resulting DataFrame.
# Assigning to combined:

# The concatenated DataFrame is assigned to the variable combined.


# In[85]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"]) #This line fits the machine learning model (model) to the training data (train) using the specified predictors (predictors) and the target variable (Target).
    preds = model.predict(test[predictors]) #This line uses the trained model (model) to make predictions on the test data (test) using the specified predictors (predictors).
    preds = pd.Series(preds, index=test.index, name="Predictions") #This line converts the array of predictions (preds) into a pandas Series.
    combined = pd.concat([test["Target"], preds], axis=1) #This line concatenates the actual target values (test["Target"]) and the predicted values (preds) into a single DataFrame (combined) along the columns (axis=1).
    return combined


# In[91]:


# start means we will take 2500 days of data (10 years of working days)
#step: The number of data points by which the training window moves forward in each iteration. Here, step=250 likely means that after each backtest iteration, the training window shifts forward by 250 data points (approximately one year of working days).
#we are training on the 10 years data and predict values for the 11 year 
def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


# In[92]:


predictions = backtest(sp500, model, predictors)


# In[95]:


predictions["Predictions"].value_counts() # how many days we predicted the market will go up or down 


# In[96]:


# Predictions
# 0    3579   # means the market will go down 3579 days
# 1    2596   # means the market will o up for 2596 days
# Name: count, dtype: int64


# In[98]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[99]:


#across 3579+2596 training days we were about 0.5288906009244992 (50%) accurate
#if we say that the market will go up, it will go actually up 50% of the time 


# In[100]:


predictions["Target"].value_counts() / predictions.shape[0]


# In[102]:


# Target
# 1    0.534737
# 0    0.465263
# Name: count, dtype: float64

# sp500 at the days we were looking at actually went up 53% of days


# In[103]:


horizons = [2,5,60,250,1000] #calculate the mean close price at the last 2,5, 60... trading days
new_predictors = []

for horizon in horizons:
    rolling_averages = sp500.rolling(horizon).mean()

ratio_column = f"Close_Ratio_{horizon}"
sp500[ratio_column] = sp500["Close"] / rolling_averages["Close"]

trend_column = f"Trend_{horizon}"
sp500[trend_column] = sp500.shift(1).rolling(horizon).sum() ["Target"]

new_predictors += [ratio_column, trend_column]


# In[105]:


sp500 = sp500.dropna() # drops rows, where there are NA values 


# In[106]:


sp500


# In[108]:


model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)


# In[109]:


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"]) #This line fits the machine learning model (model) to the training data (train) using the specified predictors (predictors) and the target variable (Target).
    preds = model.predict_proba(test[predictors]) [:,1] #Probability, that a row will be 0 or 1, probability that the start price will go up tomorrow and a probability that the start price will go down tomorrow 
    preds[preds >= .6] = 1 # if the probability will 60 or more % that the price will go up, only than it will show 1
    preds[preds < .6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions") #This line converts the array of predictions (preds) into a pandas Series.
    combined = pd.concat([test["Target"], preds], axis=1) #This line concatenates the actual target values (test["Target"]) and the predicted values (preds) into a single DataFrame (combined) along the columns (axis=1).
    return combined


# In[110]:


predictions = backtest(sp500, model, new_predictors)


# In[111]:


predictions["Predictions"].value_counts()


# In[112]:


# Predictions
# 0.0    3745 days price will go down
# 1.0    1429 days price will go up 
# Name: count, dtype: int64


# In[114]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[115]:


# when the model predicts the price will go up - 51% it will actually go up 


# In[ ]:




