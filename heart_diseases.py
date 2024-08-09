
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay,classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import sklearn.metrics as metrics

data='PATH/heart_2020_cleaned.csv'
df=pd.read_csv(data)
print(df.head(10))

if df.isnull().values.any():
    print("There are null values in the DataFrame.")
else:
    print("No null values in the DataFrame.")
print(df.shape)

if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)

info = df.describe().T.style.set_properties(**{'background-color': 'lightyellow', 'color': 'black'})
display(info)

info =df.describe(include=['O'])[1:].T.style.background_gradient(cmap='RdYlBu')
display(info)

sns.set()
df.hist(figsize=(10,10))
plt.show()

value_count = {}
for col in df.columns:
    print(col)
    data_piece  = {'val': list(df[col].value_counts().index),'sum': list(df[col].value_counts().values)}
    value_count[col] = data_piece
    display_df = pd.DataFrame(data_piece).head(10)
    display_df_styled = display_df.style.background_gradient(cmap='Blues', subset=['sum'])
    display(display_df_styled)

sns.set_theme()
colors = sns.color_palette("muted")[0:2]

pie_chart = df['HeartDisease'].value_counts().plot.pie(
    autopct='%.1f%%',
    colors=colors,
    wedgeprops=dict(width=0.5, edgecolor='w'),
    startangle=90)

plt.axis('off') 
plt.gcf().set_facecolor('none')
plt.title("Heart Disease Distribution", fontsize=18, color='red')
plt.legend(labels=df['HeartDisease'].value_counts().index, loc='upper right', bbox_to_anchor=(0.95, 0.9))
plt.show()

numerical = df.select_dtypes(include=np.float64).columns.tolist()
categorical = df.select_dtypes(exclude=np.float64).columns.tolist()

sns.set(style="whitegrid")
fig, axes = plt.subplots(nrows=len(numerical), ncols=2, figsize=(12, 2*len(numerical)))
plt.subplots_adjust(top=0.99, hspace=0.9)

for i, feature in enumerate(numerical):
    sns.histplot(df[feature], kde=True, ax=axes[i, 0], color='skyblue')
    axes[i, 0].set_title(f'{feature} - Distribution')
    sns.boxplot(x=df[feature], ax=axes[i, 1], color='salmon')
    axes[i, 1].set_title(f'{feature} - Boxplot')

plt.tight_layout()
plt.show()

def plot_charts(categorical_part):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 4))

    for i, feature in enumerate(categorical_part):
        sns.countplot(x=feature, hue='HeartDisease', data=df, ax=axes[i])
        axes[i].legend(labels=['no-disease', 'disease'], fontsize='small')
        axes[i].set_xlabel(feature, fontsize='small')  
        axes[i].set_ylabel('Count', fontsize='small') 

    plt.tight_layout()
    plt.show()

features = ['Sex', 'Race', 'GenHealth', 'AgeCategory']
categorical_part=filtered_features = [feature for feature in categorical if feature not in features]

for i in range(0,len(categorical_part),2):
    plot_charts(categorical_part[i:i+2])

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(ncols=2, nrows=2, width_ratios=[0.8, 2.2])
df = df.sort_values(by='AgeCategory')

for i, feature in enumerate(features):
    ax = fig.add_subplot(gs[i])
    sns.countplot(x=feature, hue='HeartDisease', data=df, ax=ax)
    ax.legend(labels=['no-disease', 'disease'])
    ax.set_xlabel(feature)
    ax.set_ylabel('Count')

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

sns.kdeplot(df[df["HeartDisease"] == 'No']["SleepTime"], alpha=0.5, fill=True, color="black", label="No HeartDisease", ax=axes[0])
sns.kdeplot(df[df["HeartDisease"] == 'Yes']["SleepTime"], alpha=0.5, fill=True, color="red", label="HeartDisease", ax=axes[0])
axes[0].set_xlabel("SleepTime")
axes[0].set_ylabel("Frequency")
axes[0].legend(bbox_to_anchor=(0.78, 0.98), loc=2, borderaxespad=0.)

sns.kdeplot(df[df["HeartDisease"] == 'No']["PhysicalHealth"], alpha=0.5, fill=True, color="black", label="No HeartDisease", ax=axes[1])
sns.kdeplot(df[df["HeartDisease"] == 'Yes']["PhysicalHealth"], alpha=0.5, fill=True, color="red", label="HeartDisease", ax=axes[1])
axes[1].set_xlabel("PhysicalHealth")
axes[1].set_ylabel("Frequency")
axes[1].legend(bbox_to_anchor=(0.78, 0.98), loc=2, borderaxespad=0.)

sns.kdeplot(df[df["HeartDisease"] == 'No']["MentalHealth"], alpha=0.5, fill=True, color="black", label="No HeartDisease", ax=axes[2])
sns.kdeplot(df[df["HeartDisease"] == 'Yes']["MentalHealth"], alpha=0.5, fill=True, color="red", label="HeartDisease", ax=axes[2])
axes[2].set_xlabel("MentalHealth")
axes[2].set_ylabel("Frequency")
axes[2].legend(bbox_to_anchor=(0.78, 0.98), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()

df = pd.get_dummies(df, columns=list(categorical))
x = df.loc[:, df.columns.difference(['HeartDisease_No', 'HeartDisease_Yes'])]
y = df['HeartDisease_Yes']

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

sc = StandardScaler()
le = LabelEncoder()

x = sc.fit_transform(x)
df[df.columns] = df[df.columns].apply(lambda col: le.fit_transform(col) if col.dtype == 'object' else col)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=40, shuffle=True)

print('x_train Shape : ', x_train.shape)
print('y_train Shape : ', y_train.shape)
print('x_test Shape : ', x_test.shape)
print('y_test Shape : ', y_test.shape)


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

log_reg_poly = LogisticRegression()
log_reg_poly.fit(x_train_poly, y_train)

y_pred_poly = log_reg_poly.predict(x_test_poly)
train_accuracy_poly = log_reg_poly.score(x_train_poly, y_train)
test_accuracy_poly = log_reg_poly.score(x_test_poly, y_test)

print('Accuracy score of Logistic Regression with polynomial features on training data: {}%\n'.format(train_accuracy_poly * 100))
print('Accuracy score of Logistic Regression with polynomial features on test data: {}% \n'.format(test_accuracy_poly * 100))
print('\n Logistic Regression Classification Report with polynomial features: \n', classification_report(y_test, y_pred_poly))


from sklearn.decomposition import PCA

pca = PCA(n_components=5) 
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

log_reg_pca = LogisticRegression()
log_reg_pca.fit(x_train_pca, y_train)

y_pred_pca = log_reg_pca.predict(x_test_pca)
train_accuracy_pca = log_reg_pca.score(x_train_pca, y_train)
test_accuracy_pca = log_reg_pca.score(x_test_pca, y_test)

print('Accuracy score of Logistic Regression with PCA on training data: {}%\n'.format(train_accuracy_pca * 100))
print('Accuracy score of Logistic Regression with PCA on test data: {}% \n'.format(test_accuracy_pca * 100))
print('\n Logistic Regression Classification Report with PCA: \n', classification_report(y_test, y_pred_pca))


from sklearn import tree
from sklearn.model_selection import GridSearchCV

DT = tree.DecisionTreeClassifier(random_state=0)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=DT, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_

best_DT = tree.DecisionTreeClassifier(random_state=0, **best_params)
best_DT.fit(x_train, y_train)

y_pred_tuned = best_DT.predict(x_test)
train_accuracy_tuned = best_DT.score(x_train, y_train)
test_accuracy_tuned = best_DT.score(x_test, y_test)

print('Best hyperparameters:', best_params)
print('Accuracy score of tuned Decision Tree on training data: {}%\n'.format(train_accuracy_tuned * 100))
print('Accuracy score of tuned Decision Tree on test data: {}% \n'.format(test_accuracy_tuned * 100))
print('\nTuned Decision Tree Classification Report: \n', classification_report(y_test, y_pred_tuned))


from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

RF_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=6)
RF_classifier.fit(x_train, y_train)

feature_importances = RF_classifier.feature_importances_
top_feature_indices = feature_importances.argsort()[::-1]
print("Feature Importances:")

for index in top_feature_indices:
    print(f"Feature {index + 1}: {feature_importances[index]}")

plt.figure(figsize=(10, 6))
plt.bar(range(x_train.shape[1]), feature_importances[top_feature_indices], align="center")
plt.xticks(range(x_train.shape[1]), top_feature_indices + 1)
plt.xlabel("Feature Index")
plt.ylabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.show()

y_pred = RF_classifier.predict(x_test)
train_accuracy = RF_classifier.score(x_train, y_train)
test_accuracy = RF_classifier.score(x_test, y_test)

print('Accuracy score of Random Forest on training data: {}%\n'.format(train_accuracy * 100))
print('Accuracy score of Random Forest on test data: {}% \n'.format(test_accuracy * 100))
print('\n Random Forest Classification Report: \n', classification_report(y_test, y_pred))


from sklearn import svm
from sklearn.model_selection import GridSearchCV

svm_classifier = svm.SVC()
param_grid = {
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

best_params = grid_search.best_params_
best_svm_classifier = svm.SVC(**best_params)
best_svm_classifier.fit(x_train, y_train)

y_pred_tuned = best_svm_classifier.predict(x_test)
train_accuracy_tuned = best_svm_classifier.score(x_train, y_train)
test_accuracy_tuned = best_svm_classifier.score(x_test, y_test)

print('Best hyperparameters:', best_params)
print('Accuracy score of tuned SVM on training data: {}%\n'.format(train_accuracy_tuned * 100))
print('Accuracy score of tuned SVM on test data: {}% \n'.format(test_accuracy_tuned * 100))
print('\nTuned SVM Classification Report: \n', classification_report(y_test, y_pred_tuned))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

KNN = KNeighborsClassifier()
param_grid = {
    'n_neighbors': list(range(1, 21)),  
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(estimator=KNN, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)
best_params = grid_search.best_params_

best_KNN = KNeighborsClassifier(**best_params)
best_KNN.fit(x_train, y_train)
y_pred_tuned = best_KNN.predict(x_test)

train_accuracy_tuned = best_KNN.score(x_train, y_train)
test_accuracy_tuned = best_KNN.score(x_test, y_test)

print('Best hyperparameters:', best_params)
print('Accuracy score of tuned KNN on training data: {}%\n'.format(train_accuracy_tuned * 100))
print('Accuracy score of tuned KNN on test data: {}% \n'.format(test_accuracy_tuned * 100))
print('\nTuned KNN Classification Report: \n', classification_report(y_test, y_pred_tuned))