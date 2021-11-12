# Titanic-Logistic-Regression-SVM-Decision-Tree-Random-Forest-Voting-Classifier

## Language and Machine Learning Methods:

- Python

- Logistic Regression

- Support Vector Machine

- Decision Tree

- Random Forest

- Voting Classifier

## Machine Learning Workflow

1. Define Problem
1. Specify Inputs & Outputs
1. Exploratory Data Analysis
1. Data Collection
1. Data Preprocessing
1. Data Cleaning
1. Visualization
1. Model Design, Training, and Offline Evaluation
1. Model Deployment, Online Evaluation, and Monitoring
1. Model Maintenance, Diagnosis, and Retraining

## Features

The Titanic got sunken on **April 15th, 1912**, which caused deaths of **1502 / 2224** passengers and sailors.

In this dataset, I will use the main *12 features** as follows:

Variable | Name	Description 
--- | --- 
Survived |	Survived (1) or died (0)
Pclass |	Passenger's class
Name	| Passenger's name
Sex	| Passenger's sex
Age	| Passenger's age
SibSp	| Number of siblings/spouses aboard
Parch	| Number of parents/children aboard
Ticket	| Ticket number
Fare	| Fare
Cabin	| Cabin
Embarked	| Port of embarkation

## Dataset: 

https://www.kaggle.com/c/titanic/data

## Exploratory Data Analysis(EDA)
*   Data Collection
*   Visualization
*   Data Preprocessing
*   Data Cleaning

<img src="http://s9.picofile.com/file/8338476134/EDA.png">

## Data Collection

## Visualization

**Filter unique values of features**

FInd out categorical columns (PClass) instead of continuous ones (Price).

```
print('Pclass unique values: ', df_train.Pclass.unique())
print('SibSp unique values: ', df_train.SibSp.unique())
print('Parch unique values: ', df_train.Parch.unique())
print('Sex unique values: ', df_train.Sex.unique())
print('Cabin unique values: ', df_train.Cabin.unique())
```

```
Pclass unique values:  [3 1 2]
SibSp unique values:  [1 0 3 4 2 5 8]
Parch unique values:  [0 1 2 5 3 4 6]
Sex unique values:  ['male' 'female']
Cabin unique values:  [nan 'C85' 'C123' 'E46' 'G6' 'C103' 'D56' 'A6' 'C23 C25 C27' 'B78' 'D33'
 'B30' 'C52' 'B28' 'C83' 'F33' 'F G73' 'E31' 'A5' 'D10 D12' 'D26' 'C110'
 'B58 B60' 'E101' 'F E69' 'D47' 'B86' 'F2' 'C2' 'E33' 'B19' 'A7' 'C49'
 'F4' 'A32' 'B4' 'B80' 'A31' 'D36' 'D15' 'C93' 'C78' 'D35' 'C87' 'B77'
 'E67' 'B94' 'C125' 'C99' 'C118' 'D7' 'A19' 'B49' 'D' 'C22 C26' 'C106'
 'C65' 'E36' 'C54' 'B57 B59 B63 B66' 'C7' 'E34' 'C32' 'B18' 'C124' 'C91'
 'E40' 'T' 'C128' 'D37' 'B35' 'E50' 'C82' 'B96 B98' 'E10' 'E44' 'A34'
 'C104' 'C111' 'C92' 'E38' 'D21' 'E12' 'E63' 'A14' 'B37' 'C30' 'D20' 'B79'
 'E25' 'D46' 'B73' 'C95' 'B38' 'B39' 'B22' 'C86' 'C70' 'A16' 'C101' 'C68'
 'A10' 'E68' 'B41' 'A20' 'D19' 'D50' 'D9' 'A23' 'B50' 'A26' 'D48' 'E58'
 'C126' 'B71' 'B51 B53 B55' 'D49' 'B5' 'B20' 'F G63' 'C62 C64' 'E24' 'C90'
 'C45' 'E8' 'B101' 'D45' 'C46' 'D30' 'E121' 'D11' 'E77' 'F38' 'B3' 'D6'
 'B82 B84' 'D17' 'A36' 'B102' 'B69' 'E49' 'C47' 'D28' 'E17' 'A24' 'C50'
 'B42' 'C148']
 ```
 
### Count unique values
```
print(df_train['Pclass'].value_counts())
print(df_train['SibSp'].value_counts())
print(df_train['Parch'].value_counts())
```

I will not print the Cabin's unique values since it has too many values.

### Count Plot
 
#### **Survived vs Sex**
 
 ![Survived vs Sex](https://user-images.githubusercontent.com/70437668/141410888-888864e3-28d3-412f-a2ed-7ce3417d74fe.jpg)

#### **Survived vs Pclass**

![Survived vs Pclass](https://user-images.githubusercontent.com/70437668/141410903-c0a31c03-f1c6-4313-abdc-9c61635bd489.jpg)


##### Scatter Plot

[Scatter plot](https://en.wikipedia.org/wiki/Scatter_plot) Identify correlation between 2 features


#### **Survived vs Pclass vs Fare vs Age**

![Survived vs Pclass vs Fare vs Age](https://user-images.githubusercontent.com/70437668/141410914-e5fe72d8-0f80-425b-9e75-cc88b6739c78.jpg)


#### **Survived vs Pclass vs SibSp vs Parch**

![Survived vs Pclass vs SibSp vs Parch](https://user-images.githubusercontent.com/70437668/141410926-dbe4bf72-1d24-4a42-a060-06878f2ba7a8.jpg)

### Box plot

This plot describes numerical/continuous values by their quartiles.

- https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/box-plot-review
- https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule
<img src="https://miro.medium.com/max/13500/1*2c21SkzJMf3frPXPAR_gZA.png">


**Example**

![boxplot](https://i.imgur.com/Mcw6vXv.png)


### Histogram

![Histogram](https://user-images.githubusercontent.com/70437668/141410934-7f898ef0-10da-4b11-8dfe-de04ed65bf5a.jpg)

### Correlation Heatmap

![Correlation Matrix](https://user-images.githubusercontent.com/70437668/141410947-679018ea-c750-4457-9afe-0c08bbf2fe85.jpg)

## Data Preprocessing

**Data preprocessing** is to normalize, cleanse dataset before applying algorithms to it.

Common techniques of Data preprocessing:

* Preprocess imbalanced dataset
* Preprocess dataset with NaN values
* Preprocess noise (https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba)
* Normalize dataset by Scaling (https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029) 
* Select features (https://medium.com/analytics-vidhya/feature-selection-using-scikit-learn-5b4362e0c19b)

## Data exploration

## Feature transformations

I will transform features into new types so they could bring better outcome for analysis and decision making: 

1. Name

2. Age

3. SibSp & Parch

## Feature Encoding

Apply One Hot Encoding for all categorical columns

- One hot encoding for 2 values: label 0 and 1

- One hot encoding for more than 2 values: use get_dummies

## Prepare dataset

### Get the label y

```
y = df_train.Survived
df_train = df_train.drop(columns=['Survived'])
```

### **Train / Validation Split**
```
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(df_train, y, test_size=0.3, stratify=y, shuffle=True, random_state=1612)   
print('Shape of X train', X_train.shape)
print('Shape of y train', y_train.shape)
print('Shape of X val', X_val.shape)
print('Shape of y val', y_val.shape)
```

### **Feature Scaling (MinMaxScaler)**
```
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Before using MinMaxScaler, get that column and reshape it, then transform it
fares_train = np.array(X_train['Fare']).reshape(-1, 1)
fares_val = np.array(X_val['Fare']).reshape(-1, 1)
fares_test = np.array(df_test['Fare']).reshape(-1, 1)

X_train['Fare'] = scaler.fit_transform(fares_train)
X_val['Fare']= scaler.transform(fares_val)
df_test['Fare'] = scaler.transform(fares_test)
```

## 2D Visualization

### PCA decreases dimension - Method 1

Feasible but not the best method

```
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

pca = PCA(n_components = 2) # 2D so number of components = 2
df_pca = pca.fit_transform(X_train)
```

```
plt.figure(figsize =(8, 8))
sns.scatterplot(df_pca[:,0], df_pca[:,1], hue=y_train, legend='full') # seaborn is more modern than matplotlib
```
![PCA](https://user-images.githubusercontent.com/70437668/141410964-8f089c10-7db5-4a2c-a5f7-354ffbaa2823.jpg)

```
print(pca.explained_variance_ratio_)
# Retained data is now only 0.56 + 0.16 = 72%
```

```
[0.56122445 0.16031097]
```

### T-SNE decreases dimension to 2D - Method 2

Visualize the embedded Z vector

```
from sklearn.manifold import TSNE

tsne = TSNE()
df_tsne = tsne.fit_transform(X_train)

plt.figure(figsize =(8, 8))
sns.scatterplot(df_tsne[:,0], df_tsne[:,1], hue=y_train, legend='full')
```

![TSNE](https://user-images.githubusercontent.com/70437668/141410979-17a69040-027b-4133-96ee-0d19c8231803.jpg)

## Ensemble Model

A combination of different models

### Logistic Regression

```
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train) # fit on both X train, y train
```

```
print('Accuracy on Train Set: ', logreg.score(X_train, y_train))
print('Accuracy on Validation Set: ', logreg.score(X_val, y_val))
```

```
Accuracy on Train Set:  0.8443017656500803
Accuracy on Validation Set:  0.8059701492537313
```

**Both accuracy values are not high. They can be correct on alive people, and incorrect on dead people. So I will draw Confusion Matrix to see the accuracy.**

### Support Vector Machine

```
C_values = [0.01, 0.1, 1] # 0.01 tới 10
gamma_values = [0.01, 0.1, 1]
kernel_values = ['linear', 'poly', 'rbf']

param_grid = {
    'kernel': kernel_values,
    'C': C_values,
    'gamma': gamma_values
}
```

```
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

model = SVC(random_state=1612)
grid = GridSearchCV(model, param_grid, cv=2)
grid.fit(X_train, y_train)

svc = grid.best_estimator_
```

```
print('Accuracy on Train Set: ', svc.score(X_train, y_train))
print('Accuracy on Validation Set: ', svc.score(X_val, y_val))
```

```
Accuracy on Train Set:  0.8507223113964687
Accuracy on Validation Set:  0.7910447761194029
```

### Decision Tree

```
params = {
    'criterion': ['entropy','gini'],
    'max_depth': [3,5,7],
    'min_samples_split': np.linspace(0.1, 1.0, 10), 
    'max_features':  ['auto', 'log2']
}
from sklearn.tree import DecisionTreeClassifier

decision_tree = GridSearchCV(DecisionTreeClassifier(random_state=1612), params, cv=2, n_jobs=1)
decision_tree.fit(X_train, y_train)
```

```
print('Accuracy on Train Set: ', decision_tree.score(X_train, y_train))
print('Accuracy on Validation Set: ', decision_tree.score(X_val, y_val))
```

```
Accuracy on Train Set:  0.826645264847512
Accuracy on Validation Set:  0.7761194029850746
```

### Random Forest
```
param_grid_random={'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7],
            'max_features': ['auto', 'log2'],
            'n_estimators': [100, 300, 500]}
 
from sklearn.ensemble import RandomForestClassifier 

random_forest = GridSearchCV(RandomForestClassifier(random_state=1612), params, cv=2, n_jobs=1)
random_forest.fit(X_train, y_train)
```

```
print('Accuracy on Train Set: ', random_forest.score(X_train, y_train))
print('Accuracy on Validation Set: ', random_forest.score(X_val, y_val))
```

```
Accuracy on Train Set:  0.8491171749598716
Accuracy on Validation Set:  0.7985074626865671
```

### Voting Classifier

voting = {'hard', 'soft'}

If 'hard', uses predicted class labels for majority rule voting. Else if 'soft', predicts the class label based on the argmax of the sums of the predicted probabilities, which is recommended for an ensemble of well-calibrated classifiers.

```
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(estimators=[
                              ('Logistic Regression', logreg), 
                              ('SVM', svc), 
                              ('Decision Tree', decision_tree)],
                            voting='hard',
                            n_jobs=-1) # hard: label which has many ones can be returned # ('Random Forest', random_forest)

ensemble.fit(X_train, y_train)
```

```
# Use score() function on 2 Sets
print('Accuracy on Train Set: ', ensemble.score(X_train, y_train))
print('Accuracy on Validation Set: ', ensemble.score(X_val, y_val))
```

```
Accuracy on Train Set:  0.8491171749598716
Accuracy on Validation Set:  0.7873134328358209
```

### **Confusion Matrix**
```
# 1. Import confusion matrix from sklearn
# 2. Use confusion_matrix to draw a heatmap
# 3. if your heatmap show 8e+2 numbers. Insise heatmap() function, put a parameter fmt='.1f'  
from sklearn.metrics import confusion_matrix
y_pred = ensemble.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt='.1f')
plt.title('Confusion Matrix')
```

![Confusion Matrix](https://user-images.githubusercontent.com/70437668/141410988-a08dd655-f8df-4a1f-bde8-ccafed7d687e.jpg)
