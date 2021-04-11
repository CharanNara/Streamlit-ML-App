import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
st.title("Streamlit Basic WebApp")
st.write("""
which is the best classifier?
         """)
dataset = st.sidebar.selectbox("Select Dataset",("IRIS","WINE DATASET"))
classifier = st.sidebar.selectbox("Select Classifier",("KNN","RANDOM FOREST","SVM"))


def get_dataset(dataset):
    if dataset == "IRIS":
        data = datasets.load_iris()
    elif dataset == "WINE DATASET":
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X,y
X,y = get_dataset(dataset)
st.write("Shape of Dataset: ",X.shape)
st.write("noof classes: ",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2,15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

params = add_parameter_ui(classifier)

def get_classifier(clf_name,params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C = params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],n_estimators=params["n_estimators"])
    return clf

clf = get_classifier(classifier, params)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state=1234)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test,y_pred)
st.write(f"Classifier = {classifier}")
st.write(f"Accuracy = {acc}")
