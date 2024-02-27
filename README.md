## TASK NO : 3
# CUSTOMER CHURN PREDICTION

![t3](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/63965a07-dc79-4af0-acd8-83ad94223937)

## CODE

```
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```
### Read the given dataset
```
df = pd.read_csv("Churn_Modelling.csv")
df.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/22135a02-15ff-411e-96c3-0d7e067db542)

```
df.info()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/88312872-c110-48ff-813a-99bc22cdde16)

```
df.describe()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/07b3885e-bc07-4435-add6-bf7482d904fd)

```
#Data Analysing
fig,axb = plt.subplots(ncols=2,nrows=1,figsize=(16, 8))

#Gender Distribution
explode = [0.1, 0.1]
df.groupby('Gender')['Exited'].count().plot.pie(explode=explode, autopct="%1.1f%%",ax=axb[0]);

ax = sns.countplot(x="Gender", hue="Exited", data=df,ax=axb[1])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.title("Distribution of Gender with Exited Status")
plt.xlabel("Gender")
plt.ylabel("Count")

# Show the plot
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/035405e5-0221-4b91-9cd0-5c4b39b1e619)

```
# Exited Counts Pie Chart
is_Exited = df["Exited"].value_counts()
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(is_Exited, labels=["No", "Yes"], autopct="%0.0f%%")
plt.title("Is_Exited Counts")

# Distribution of Geography Pie Chart
plt.subplot(1, 2, 2)
geography_counts = df['Geography'].value_counts()
plt.pie(geography_counts, labels=geography_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal') 
plt.title('Distribution of Geography')

plt.tight_layout() 
plt.show()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/13ae1f60-aba9-4cc7-9b68-bf0a2141e3a8)

```
#Data Cleaning
df = df.drop(['RowNumber', 'Surname', 'CustomerId'], axis= 1)
#preprocessing
df['Balance'] = df['Balance'].astype(int)
df['EstimatedSalary'] = df['EstimatedSalary'].astype(int)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Geography'] = le.fit_transform(df['Geography'])

df.head()
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/fb980f94-a46d-44c1-a83c-ede2a11f6187)

```
No_class = df[df["Exited"]==0]
yes_class = df[df["Exited"]==1]
s = pd.concat([yes_class, No_class], axis=0)
X = s.drop("Exited", axis=1)
y = s["Exited"]
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)

#Random Forest Classifier 
rf_model=RandomForestClassifier()
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)
print(classification_report(y_test, y_pred))
rf_accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(rf_accuracy * 100))

#Logistic regression
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
print(classification_report(y_test, y_pred))
lr_accuracy = accuracy_score(y_pred,y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(lr_accuracy*100))

#Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb.fit(x_train, y_train)
y_pred = gb.predict(x_test)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
gb_accuracy = accuracy_score(y_test, y_pred)
print('XGBoost model accuracy is: {:.2f}%'.format(gb_accuracy * 100))

#SVM
svm_model = LinearSVC()
svm_model.fit(x_train, y_train)
predict = svm_model.predict(x_test)
print(classification_report(y_test, predict))

svm_accuracy = accuracy_score(predict,y_test)
print('SVC model accuracy is: {:.2f}%'.format(svm_accuracy*100))
```
### Random Forest Algorithm
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/a9ef4d28-f332-4a79-89b9-3f7ba26914fc)

### Logistic Regression
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/71896f97-6759-464d-8199-283c929d62da)

### Gradient Boosting
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/a8273ca7-7cbd-42fb-9aa6-149a70405e4f)

### SVC Algorithm
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/fd2b1952-3b1e-4065-9088-304b95201f50)

```
#Comparing accuracies
Algorithms = ['Gradient Boosting', 'Random Forest','Logistic Regression', 'SVC']
accuracy = [gb_accuracy, rf_accuracy, lr_accuracy, svm_accuracy]
per_acc=[]
for i in accuracy:
    per_acc.append(i*100)
FinalResult=pd.DataFrame({'Accuracy':per_acc,'Algorithm':Algorithms})

FinalResult
```
![image](https://github.com/AnnBlessy/codsoft_taskno.3/assets/119477835/985edacb-52c6-437e-8461-f752f4d65472)
