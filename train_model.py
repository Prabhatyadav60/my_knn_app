import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import pickle


dataset = pd.read_csv('E:\LUV\Project_KNN\Social_Network_Ads.csv')  

X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2)
knn_classifier.fit(X_train, y_train)


pickle.dump(knn_classifier, open('knn_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

print("Model and Scaler saved successfully!")
