import pandas as pd
s = pd.read_csv("social_media_usage.csv")
print(s.shape)
s.head(10)


##Import Packages
import pandas as pd
import numpy as np
import streamlit as st
##import matplotlib.pyplot as plt
##import seaborn as sns
##import altair as alt
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


##Filter Warnings
import warnings
warnings.filterwarnings("ignore")

def clean_sm(x):
    x=np.where((x==1),1, 0)
    return x

# * Create a new dataframe called "ss". 
# * The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn
# * Use the following features: 
#     + income (ordered numeric from 1 to 9, above 9 considered missing)
#     + education (ordered numeric from 1 to 8, above 8 considered missing)
#     + parent (binary)
#     + married (binary)
#     + female (binary)
#     + age (numeric, above 98 considered missing)
# * Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.


ss = pd.DataFrame({
    "sm_li": s["web1h"].apply(clean_sm),
    "income": np.where(s["income"]>9, np.nan, s["income"]),
    "education": np.where(s['educ2']>8, np.nan, s["educ2"]),
    "parent": np.where(s['par']==1, 1, 0),
    "married": np.where(s['marital'] ==1, 1, 0),
    "female": np.where(s['gender']==2,1, 0),
    "age": np.where(s['age'] >97, np.nan, s["age"])})

ss = ss.dropna()

# * **Target**: sm_li (LinkedIn User)
#     + Is a LinkedIn user (=1)
#     + Is not a LinkedIn user (=0)
# * **Features**:
#     + income (1 'Less than 10,000' -> 9 'Greater than 150,000')
#     + education (1 'Less than high school' -> 8 'Postgraduate or professional degree')
#     + parent (binary)
#     + married (binary)
#     + female (binary)
#     + age (numeric)


y = ss["sm_li"]
x = ss.drop("sm_li", axis=1)


xtrain, xtest, ytrain, ytest = train_test_split(x, y,
                                                random_state=104,  
                                                test_size=0.20, 
                                                stratify=y,
                                                shuffle=True) 

# **xtrain**: this object contains 80% (1008 rows, 6 columns) of the x features that will be used to build and train the model
# 
# **xtest**: This object contains 20% (252 rows, 6 columns) of the x features that will be used to test the output of the model on unseen data
# 
# **ytrain**: this object contains 80% (1008 rows, 1 column) of the y, target that will be used to build and train the model
# 
# **ytest**: This object contains 20% (252 rows, 1 column) of the y, target that will be used to test the output of the model on unseen data to evaluate performance

# #### Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.


Lr1 = LogisticRegression(class_weight='balanced')


Lr1.fit(xtrain, ytrain)

# #### Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# ####  Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents


st.title('Patrick's LinkedIn User Predictions')
st.caption('Configure parameters to predict if someone is a LinkedIn user')

Income_options = ["Less than $10,000", "10 to under $20,000", "20 to under $30,000", "30 to under $40,000", "40 to under $50,000", "50 to under $75,000", "75 to under $100,000", "100 to under $150,000", "$150,000 or more?"]

Income_map = {opt: idx + 1 for idx, opt in enumerate(Income_options)}

Income_box = st.selectbox("Select Income Level", options=Income_options)

income = Income_map.get(Income_box, 0)

Education_options = ["1 Less than high school (Grades 1-8 or no formal schooling)", "High school incomplete (Grades 9-11 or Grade 12 with NO diploma)", "High school graduate (Grade 12 with diploma or GED certificate)", "Some college, no degree (includes some community college)", "Two-year associate degree from a college or university", "Four-year college or university degree/Bachelorâs degree (e.g., BS, BA, AB)", "Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)", "Postgraduate or professional degree, including masterâs, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)"]

Education_map = {opt: idx + 1 for idx, opt in enumerate(Education_options)}

Education_box = st.selectbox("Select Education Level", options=Education_options)

education = Education_map.get(Education_box, 0)

binary_options_df = pd.DataFrame({
    'parent': [1,0],
    'married': [1,0],
    'female': [1,0],
})
#     parent (binary)

parent = st.selectbox(
    'Are they a parent? 1=Yes, 0=No',
    ('1', '0'))

st.write('You selected:', parent)

#     married (binary)
married = st.selectbox(
    'Are they married? 1=Yes, 0=No',
    ('1', '0'))

st.write('You selected:', married)

#     female (binary)
female = st.selectbox(
    'Are they female? 1=Yes, 0=No',
    ('1', '0'))

st.write('You selected:', female)
#     age (numeric)

age= st.slider(label="Select Age",
    min_value=1,
    max_value=97,
    value=30)


input_data = pd.DataFrame([[income, education, parent, married, female, age]],
    columns=["income", "education", "parent", "married", "female", "age"])

pred_result = Lr1.predict(input_data)
pred_probability = Lr1.predict_proba(input_data)[:,1]
pred_percent = pred_probability*100

if pred_result ==1:
    pred_label = "be a LinkedIn user"
else: 
    pred_label = "not be a LinkedIn user"

container = st.container(border=True)
container.write(f"This person is predected to**{pred_label}** with {pred_percent[0]}% probability")

