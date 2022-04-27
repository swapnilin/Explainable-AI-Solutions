
import pandas as pd


#Reading the data
#we will save a raw copy of the training set
raw = pd.read_csv('train.csv')

#Importing the training data
train = pd.read_csv('train.csv')


#Missing value imputation
train['Gender'].fillna(train['Gender'].mode()[0], inplace=True)
train['Married'].fillna(train['Married'].mode()[0], inplace=True)
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True)
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True)

train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


#rename Some features for better interpretability
train = train.rename(columns = {'Education':'Graduate','Gender':'Gender_Male','Loan_Status':'Eligible'})

#Data Prep

#We' label encode dependents
from sklearn import preprocessing
encode = preprocessing.LabelEncoder()

train['Dependents'] = encode.fit_transform(train.Dependents.values)
train['Credit_History'] = encode.fit_transform(train.Credit_History.values)
train['Married'] = encode.fit_transform(train.Married.values)
train['Gender_Male'] = encode.fit_transform(train.Gender_Male.values)
train = train.replace({'Graduate':{'Graduate':1,'Not Graduate':0}})
train = train.replace({'Self_Employed':{'Yes':1,'No':0}})
train = train.replace({'Eligible':{'Y':1,'N':0}})


#Apply One-Hot Encoding
train = pd.get_dummies(train, columns=['Property_Area'])
train = pd.get_dummies(train, columns=['Dependents'])


#saving this procewssed dataset
#train.to_csv('Processed_Credit_Data.csv', index=None)

#Drop Loan ID
train = train.drop(['Loan_ID'], axis=1)


#rename some column names for interpretability
train = train.rename(columns = {'Property_Area_Rural':'Rural_Property', 'Property_Area_Urban':'Urban_Property', 'Property_Area_Semiurban':'Semiurban_Property'})


#Data Partition


#Seperate the target vaiable
x = train.drop('Eligible',1)
y = train.Eligible


#splitting the data in training and testing set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=123)


#Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lrmodel = LogisticRegression()
lrmodel.fit(x_train, y_train)


#Let’s predict the 'Eligibility' for testing set and calculate its accuracy.
lrpred = lrmodel.predict(x_test)
accuracy_score(y_test,lrpred)


#Import all the required libraries from Dashboards
from explainerdashboard import ClassifierExplainer, ExplainerDashboard
from explainerdashboard import 

#fit model
lrmodel.fit(x_train, y_train)

#save the model
import pickle
pickle.dump(lrmodel, open('lrmodel.pkl','wb'))

x_test.iloc[0]

#Explainer Instance
#lrSHAPexplainer = ClassifierExplainer(lrmodel, x_test, y_test)
lrSHAPexplainer = ClassifierExplainer(lrmodel, x_test)
lrinference =  

#Dashboard Configuration
db = ExplainerDashboard(lrSHAPexplainer, title="Loan Eligibility",
                    whatif=True, # you can switch off tabs with bools
                    shap_interaction=False,
                    decision_trees=False)

db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
