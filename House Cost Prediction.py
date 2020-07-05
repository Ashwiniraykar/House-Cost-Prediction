
# coding: utf-8

# In[81]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn import datasets


# In[82]:


ndata = pd.read_csv("DC_Properties.csv")


# In[83]:


ndata.head()


# In[84]:


ndata.describe()


# In[85]:


ndata.info()


# In[86]:


# count the number of NaN values in each column
print(ndata.isnull().sum())


# In[87]:


ndata['SALEDATE'] = pd.to_datetime(ndata['SALEDATE'])
ndata['SALEYEARR']=ndata['SALEDATE'].dt.year
ndata['SALEMONTHH']=ndata['SALEDATE'].dt.month
ndata['SALEDAYY']=ndata['SALEDATE'].dt.day
ndata['SALEDATE']
ndata['SALEDATE'].dropna();


# In[88]:


data_new=ndata.drop(['CENSUS_BLOCK','CENSUS_TRACT','ASSESSMENT_SUBNBHD','ASSESSMENT_NBHD','NATIONALGRID','STATE'
                    ,'CITY','FULLADDRESS','CMPLX_NUM','QUADRANT','Y','X','WARD','LIVING_GBA','SOURCE',
                    'GIS_LAST_MOD_DTTM','USECODE','INTWALL','EXTWALL','STYLE','BLDG_NUM','SALE_NUM'
                    ,'NUM_UNITS','HEAT','HF_BATHRM','Unnamed: 0'],axis=1)


# In[89]:


pd.get_dummies(ndata['ROOF'])
pd.get_dummies(ndata['AC'])
pd.get_dummies(ndata[['QUALIFIED']])
pd.get_dummies(ndata['GRADE'])
pd.get_dummies(ndata['STRUCT'])
pd.get_dummies(ndata[['CNDTN']])


# In[90]:


roof = pd.get_dummies(data_new['ROOF'],drop_first=True)
Qualified_N = pd.get_dummies(data_new['QUALIFIED'],drop_first=True)
AC_N = pd.get_dummies(data_new['AC'],drop_first=True)


# In[91]:


grade = pd.get_dummies(data_new['GRADE'],drop_first=True)
struct = pd.get_dummies(data_new['STRUCT'],drop_first=True)
cndtn = pd.get_dummies(data_new[['CNDTN']],drop_first=True)


# In[92]:


data_new = pd.concat([data_new,roof,AC_N,Qualified_N,grade,struct,cndtn],axis=1)


# In[93]:


data_new.drop(['ROOF','AC','QUALIFIED','CNDTN','GRADE','STRUCT'],axis=1,inplace=True)


# In[94]:


data_new[data_new.PRICE.isnull()]
data_new1 = data_new.dropna(how='any')


# In[95]:


data_new1.info()


# In[96]:


print(data_new1.isnull().sum())


# In[97]:


x=data_new1[['LANDAREA','BATHRM','ROOMS','KITCHENS','FIREPLACES','ZIPCODE','GBA',
     'LONGITUDE','LATITUDE','EYB','YR_RMDL','AYB','Metal- Sms', 'Neopren', 'Comp Shingle','Y','N',
     'BEDRM','Metal- Cpr','Typical', 'Slate', 'Composition Ro', 'Metal- Pre', 'Shingle',
       'Concrete', 'Shake', 'Clay Tile', 'Water Proof', 'Concrete Tile',
       'Wood- FS','Very Good', 'Average', 'Good Quality', 'Excellent',
       'Average', 'Superior', 'Fair Quality', 'Exceptional-D',
       'Exceptional-C', 'Low Quality', 'Exceptional-A', 'Exceptional-B',
       'No Data','Row Inside', 'Semi-Detached', 'Single', 'Row End', 'Multi',
       'Town Inside', 'Town End', 'Vacant Land', 'Very Good', 'Average', 'Excellent','SALEYEARR','SALEMONTHH','SALEDAYY']]
y=data_new1['PRICE']


# In[98]:


#data.drop(['EXTWALL','INTWALL','WARD','QUADRANT'],1, inplace=True)
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
data_new1.convert_objects(convert_numeric=True)
data_new1.fillna(0, inplace=True)
def handle_non_numerical_data(data_new1):
    columns = data_new1.columns.values
    
    for column in columns:
        text_digit_vals = {} 
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if data_new1[column].dtype!= np.int64 and data_new1[column].dtype!=np.float64:
            column_contents = data_new1[column].values.tolist() 
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            data_new1[column] = list(map(convert_to_int, data_new1[column]))
    return data_new1
#data = handle_non_numerical_data(data_new1)
#print(ndata.head())


# In[99]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=49)


# In[100]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[101]:


lm.fit(x_train,y_train)


# In[102]:


lm.score(x_test,y_test)


# In[103]:


cl = y.copy()
cl1 = cl.max()
break_data=cl1/10
break_data


# In[104]:


for i in range (len(cl)):
    if(cl.iloc[i]<= break_data):
        cl.iloc[i] = 0
        print('0')
    elif((cl.iloc[i]> break_data)& (cl.iloc[i] <= 2*break_data)):
        cl.iloc[i] = 1
        print('1')
    elif((cl.iloc[i]>2*break_data)& (cl.iloc[i] <= 3*break_data)):
        cl.iloc[i] = 2
        print('2')
    elif((cl.iloc[i]>3*break_data)& (cl.iloc[i] <= 4*break_data)):
        cl.iloc[i] = 3
        print('3')
    elif((cl.iloc[i]> 4*break_data)& (cl.iloc[i] <= 5*break_data)):
        cl.iloc[i] = 4
        print('4')
    elif((cl.iloc[i]> 5*break_data)& (cl.iloc[i] <=6*break_data)):
        cl.iloc[i] = 5
        print('5')
    elif((cl.iloc[i]> 6*break_data)& (cl.iloc[i] <=7*break_data)):
        cl.iloc[i] = 6
        print('6')
    elif((cl.iloc[i]> 7*break_data)& (cl.iloc[i] <=8*break_data)):
        cl.iloc[i] = 7
        print('7')
    elif((cl.iloc[i]> 8*break_data)& (cl.iloc[i] <= 9*break_data)):
        cl.iloc[i] = 8
        print('8')
    else:
        cl.iloc[i]
        print('9')


# ## Logistic Regression using sklearn library

# In[106]:


from sklearn.model_selection import train_test_split
x_train, x_test, cl_train, cl_test = train_test_split(x, cl, test_size=0.2, random_state=42)


# In[107]:


x_train.shape, cl_train.shape


# In[108]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()


# In[109]:


logr.fit(x_train,cl_train)


# In[110]:


cl_pred_class=logr.predict(x_test)


# In[111]:


from sklearn.metrics import (accuracy_score, f1_score, precision_score,
recall_score, classification_report, confusion_matrix,r2_score)

result= accuracy_score(cl_test,cl_pred_class)
result


# ## Logistic Regression using Own Implementation

# In[113]:


def sigmoid(x):
    return 1/(1+(np.exp(-x)))
def cost(X, Y, W, m):
    return  -(1./m) * np.sum(
        (np.multiply(Y, np.log(sigmoid(np.dot(W,X))))) +
        (np.multiply((1-Y),np.log(1-sigmoid(np.dot(W,X))))))


# In[114]:


def logisticRegression(X1, Y1, alpha, iterations):
    X=np.c_[np.ones((len(Y1), 1)), X1]
    Y=np.array(Y1).reshape(1,len(Y1))
    X=X.T
    cost_list = []
    m = np.size(X, axis=1)
    W = np.random.random((1, X.shape[0]))
    for i in range(iterations):
        W = W - (alpha/m) * np.dot((sigmoid(np.dot(W,X)) - Y), X.T)
        cost_ = cost(X, Y, W, m)
        cost_list.append([i, cost_])
    return W, cost_list
        


# In[118]:


y0 = cl_train.copy()
y0[cl_train==0]=1
y0[cl_train==1]=0
y0[cl_train==2]=0
y0[cl_train==3]=0
y0[cl_train==4]=0
y0[cl_train==5]=0
y0[cl_train==6]=0
y0[cl_train==7]=0
y0[cl_train==8]=0
y0[cl_train==9]=0
W0, cost_list = logisticRegression(x_train, y0, 0.1, 10000)
print('W0 =', W0)
#for i in range(0, len(cost_list), 100):
#    plt.plot(cost_list[i][0], cost_list[i][1], 'bx')
#plt.show()


# In[119]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W0,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y0_test = cl_test.copy()
y0_test[cl_test==0]=1
y0_test[cl_test==1]=0
y0_test[cl_test==2]=0
y0_test[cl_test==3]=0
y0_test[cl_test==4]=0
y0_test[cl_test==5]=0
y0_test[cl_test==6]=0
y0_test[cl_test==7]=0
y0_test[cl_test==8]=0
y0_test[cl_test==9]=0
print("Accuracy class 0 =  ", sum(y0_test == predictions[0]) / x_test.shape[0] )


# In[120]:


y1 = cl_train.copy()
y1[cl_train==0]=0
y1[cl_train==1]=1
y1[cl_train==2]=0
y1[cl_train==3]=0
y1[cl_train==4]=0
y1[cl_train==5]=0
y1[cl_train==6]=0
y1[cl_train==7]=0
y1[cl_train==8]=0
y1[cl_train==9]=0
W1, cost_list = logisticRegression(x_train, y1, 0.1, 10000)
print('W1 =', W1)


# In[121]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W1,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y1_test = cl_test.copy()
y1_test[cl_test==0]=0
y1_test[cl_test==1]=1
y1_test[cl_test==2]=0
y1_test[cl_test==3]=0
y1_test[cl_test==4]=0
y1_test[cl_test==5]=0
y1_test[cl_test==6]=0
y1_test[cl_test==7]=0
y1_test[cl_test==8]=0
y1_test[cl_test==9]=0
print("Accuracy class 1 = ", sum(y1_test == predictions[0]) / x_test.shape[0])


# In[122]:


y2 = cl_train.copy()
y2[cl_train==0]=0
y2[cl_train==1]=0
y2[cl_train==2]=1
y2[cl_train==3]=0
y2[cl_train==4]=0
y2[cl_train==5]=0
y2[cl_train==6]=0
y2[cl_train==7]=0
y2[cl_train==8]=0
y2[cl_train==9]=0
W2, cost_list = logisticRegression(x_train,y2, 0.1, 10000)
print('W2 =', W2)


# In[123]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W2,X)
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y2_test = cl_test.copy()
y2_test[cl_test==0]=0
y2_test[cl_test==1]=0
y2_test[cl_test==2]=1
y2_test[cl_test==3]=0
y2_test[cl_test==4]=0
y2_test[cl_test==5]=0
y2_test[cl_test==6]=0
y2_test[cl_test==7]=0
y2_test[cl_test==8]=0
y2_test[cl_test==9]=0
print("Accuracy class 2 = ", sum(y2_test == predictions[0]) / x_test.shape[0] )


# In[124]:


y3 = cl_train.copy()
y3[cl_train==0]=0
y3[cl_train==1]=0
y3[cl_train==2]=0
y3[cl_train==3]=1
y3[cl_train==4]=0
y3[cl_train==5]=0
y3[cl_train==6]=0
y3[cl_train==7]=0
y3[cl_train==8]=0
y3[cl_train==9]=0
W3, cost_list = logisticRegression(x_train, y3, 0.1, 10000)
print('W3 =', W3)


# In[125]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W3,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y3_test = cl_test.copy()
y3_test[cl_test==0]=0
y3_test[cl_test==1]=0
y3_test[cl_test==2]=0
y3_test[cl_test==3]=1
y3_test[cl_test==4]=0
y3_test[cl_test==5]=0
y3_test[cl_test==6]=0
y3_test[cl_test==7]=0
y3_test[cl_test==8]=0
y3_test[cl_test==9]=0
print("Accuracy class 3 = ", sum(y3_test == predictions[0]) / x_test.shape[0] )


# In[126]:


y4 = cl_train.copy()
y4[cl_train==0]=0
y4[cl_train==1]=0
y4[cl_train==2]=0
y4[cl_train==3]=0
y4[cl_train==4]=1
y4[cl_train==5]=0
y4[cl_train==6]=0
y4[cl_train==7]=0
y4[cl_train==8]=0
y4[cl_train==9]=0
W4, cost_list = logisticRegression(x_train, y4, 0.1, 10000)
print('W4 =', W4)


# In[127]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W4,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y4_test = cl_test.copy()
y4_test[cl_test==0]=0
y4_test[cl_test==1]=0
y4_test[cl_test==2]=0
y4_test[cl_test==3]=0
y4_test[cl_test==4]=1
y4_test[cl_test==5]=0
y4_test[cl_test==6]=0
y4_test[cl_test==7]=0
y4_test[cl_test==8]=0
y4_test[cl_test==9]=0
print("Accuracy class 4 = ", sum(y4_test == predictions[0]) / x_test.shape[0] )


# In[128]:


y5 = cl_train.copy()
y5[cl_train==0]=0
y5[cl_train==1]=0
y5[cl_train==2]=0
y5[cl_train==3]=0
y5[cl_train==4]=0
y5[cl_train==5]=1
y5[cl_train==6]=0
y5[cl_train==7]=0
y5[cl_train==8]=0
y5[cl_train==9]=0
W5, cost_list = logisticRegression(x_train, y5, 0.1, 10000)
print('W5 =', W5)


# In[129]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W5,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y5_test = cl_test.copy()
y5_test[cl_test==0]=0
y5_test[cl_test==1]=0
y5_test[cl_test==2]=0
y5_test[cl_test==3]=0
y5_test[cl_test==4]=0
y5_test[cl_test==5]=1
y5_test[cl_test==6]=0
y5_test[cl_test==7]=0
y5_test[cl_test==8]=0
y5_test[cl_test==9]=0
print("Accuracy class 5 = ", sum(y5_test == predictions[0]) / x_test.shape[0] )


# In[130]:


y6 = cl_train.copy()
y6[cl_train==0]=0
y6[cl_train==1]=0
y6[cl_train==2]=0
y6[cl_train==3]=0
y6[cl_train==4]=0
y6[cl_train==5]=0
y6[cl_train==6]=1
y6[cl_train==7]=0
y6[cl_train==8]=0
y6[cl_train==9]=0
W6, cost_list = logisticRegression(x_train, y6, 0.1, 10000)
print('W6 =', W6)


# In[131]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W6,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y6_test = cl_test.copy()
y6_test[cl_test==0]=0
y6_test[cl_test==1]=0
y6_test[cl_test==2]=0
y6_test[cl_test==3]=0
y6_test[cl_test==4]=0
y6_test[cl_test==5]=0
y6_test[cl_test==6]=1
y6_test[cl_test==7]=0
y6_test[cl_test==8]=0
y6_test[cl_test==9]=0
print("Accuracy class 6 = ", sum(y6_test == predictions[0]) / x_test.shape[0] )


# In[132]:


y7 = cl_train.copy()
y7[cl_train==0]=0
y7[cl_train==1]=0
y7[cl_train==2]=0
y7[cl_train==3]=0
y7[cl_train==4]=0
y7[cl_train==5]=0
y7[cl_train==6]=0
y7[cl_train==7]=1
y7[cl_train==8]=0
y7[cl_train==9]=0
W7, cost_list = logisticRegression(x_train, y7, 0.1, 10000)
print('W7 =', W7)


# In[133]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W7,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y7_test = cl_test.copy()
y7_test[cl_test==0]=0
y7_test[cl_test==1]=0
y7_test[cl_test==2]=0
y7_test[cl_test==3]=0
y7_test[cl_test==4]=0
y7_test[cl_test==5]=0
y7_test[cl_test==6]=0
y7_test[cl_test==7]=1
y7_test[cl_test==8]=0
y7_test[cl_test==9]=0
print("Accuracy class 7 = ", sum(y7_test == predictions[0]) / x_test.shape[0] )


# In[134]:


y8 = cl_train.copy()
y8[cl_train==0]=0
y8[cl_train==1]=0
y8[cl_train==2]=0
y8[cl_train==3]=0
y8[cl_train==4]=0
y8[cl_train==5]=0
y8[cl_train==6]=0
y8[cl_train==7]=0
y8[cl_train==8]=1
y8[cl_train==9]=0
W8, cost_list = logisticRegression(x_train, y8, 0.1, 10000)
print('W8 =', W8)


# In[135]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W8,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y8_test = cl_test.copy()
y8_test[cl_test==0]=0
y8_test[cl_test==1]=0
y8_test[cl_test==2]=0
y8_test[cl_test==3]=0
y8_test[cl_test==4]=0
y8_test[cl_test==5]=0
y8_test[cl_test==6]=0
y8_test[cl_test==7]=0
y8_test[cl_test==8]=1
y8_test[cl_test==9]=0
print("Accuracy class 8 = ", sum(y8_test == predictions[0]) / x_test.shape[0] )


# In[136]:


y9 = cl_train.copy()
y9[cl_train==0]=0
y9[cl_train==1]=0
y9[cl_train==2]=0
y9[cl_train==3]=0
y9[cl_train==4]=0
y9[cl_train==5]=0
y9[cl_train==6]=0
y9[cl_train==7]=0
y9[cl_train==8]=0
y9[cl_train==9]=1
W9, cost_list = logisticRegression(x_train, y9, 0.1, 10000)
print('W9 =', W9)


# In[137]:


X=np.c_[np.ones((x_test.shape[0], 1)), x_test].T
predictions = np.dot(W9,X )
predictions[predictions < 0] = 0
predictions[predictions >0] = 1
y9_test = cl_test.copy()
y9_test[cl_test==0]=0
y9_test[cl_test==1]=0
y9_test[cl_test==2]=0
y9_test[cl_test==3]=0
y9_test[cl_test==4]=0
y9_test[cl_test==5]=0
y9_test[cl_test==6]=0
y9_test[cl_test==7]=0
y9_test[cl_test==8]=0
y9_test[cl_test==9]=1
print("Accuracy class 9 = ", sum(y9_test == predictions[0]) / x_test.shape[0] )


# In[138]:


def distance(W, x):
    num = np.dot(W, x)
    deno = np.sqrt(np.dot(W, W.T))
    return num/deno


# In[139]:


C=np.c_[np.ones((x_test.shape[0], 1)), x_test].T

prediction0 = np.dot(W0, C)
prediction1 = np.dot(W1, C)
prediction2 = np.dot(W2, C)
prediction3 = np.dot(W3, C)
prediction4 = np.dot(W4, C)
prediction5 = np.dot(W5, C)
prediction6 = np.dot(W6, C)
prediction7 = np.dot(W7, C)
prediction8 = np.dot(W8, C)
prediction9 = np.dot(W9, C)
prediction0[prediction0 < 0] = 0
prediction0[prediction0 > 0] = 1
prediction1[prediction1 < 0] = 0
prediction1[prediction1 > 0] = 1
prediction2[prediction2 < 0] = 0
prediction2[prediction2 > 0] = 1
prediction3[prediction3 < 0] = 0
prediction3[prediction3 > 0] = 1
prediction4[prediction4 < 0] = 0
prediction4[prediction4 > 0] = 1
prediction5[prediction5 < 0] = 0
prediction5[prediction5 > 0] = 1
prediction6[prediction6 < 0] = 0
prediction6[prediction6 > 0] = 1
prediction7[prediction7 < 0] = 0
prediction7[prediction7 > 0] = 1
prediction8[prediction8 < 0] = 0
prediction8[prediction8 > 0] = 1
prediction9[prediction9 < 0] = 0
prediction9[prediction9 > 0] = 1
p=np.r_[prediction0,prediction1,prediction2,prediction3,prediction4,prediction5,prediction6,prediction7,prediction8,prediction9]

dist=np.r_[distance(W0, C),distance(W1, C),distance(W2, C),distance(W3, C),distance(W4, C),distance(W5, C),distance(W6, C),distance(W7, C),distance(W8, C),distance(W9, C)]
prediction=np.argmax(dist,0)

print("Accuracy : ", sum(cl_test == prediction) / x_test.shape[0] )


# ## NEURAL NETWORK

# In[141]:


from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=0.001)


# In[142]:


from sklearn.model_selection import train_test_split
x_train, x_test, cl_train, cl_test = train_test_split(x, cl, test_size=0.25, random_state=99)


# In[143]:


ppn.fit(x_train,cl_train)
predict_NN = ppn.predict(x_test)
print(predict_NN.shape)
print(predict_NN)


# In[144]:


from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, classification_report, confusion_matrix,r2_score
score=accuracy_score(cl_test, predict_NN)
print('Accuracy: ',score)

