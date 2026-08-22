import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Customer Churn Analytics",page_icon="📉",layout="wide")
df=pd.read_csv("customer_churn.csv")
st.title("📉 Customer Churn Prediction & Retention Analytics")
st.caption("Data cleaning • SQL-ready analytics • ML prediction • business insights")

c1,c2,c3,c4=st.columns(4)
c1.metric("Customers",len(df))
c2.metric("Churn Rate",f"{df.churn.mean()*100:.1f}%")
c3.metric("Avg Monthly Charges",f"${df.monthly_charges.mean():.2f}")
c4.metric("Avg Tenure",f"{df.tenure_months.mean():.1f} mo")

st.subheader("Churn Analysis")
a,b=st.columns(2)
with a:
    x=df.groupby("contract",as_index=False).churn.mean()
    x["churn"]=x.churn*100
    st.plotly_chart(px.bar(x,x="contract",y="churn",labels={"churn":"Churn %","contract":""}),use_container_width=True)
with b:
    st.plotly_chart(px.histogram(df,x="tenure_months",color=df.churn.map({0:"Stayed",1:"Churned"}),barmode="overlay",labels={"color":"Status"}),use_container_width=True)

st.subheader("Machine Learning Model")
features=["age","tenure_months","monthly_charges","support_calls","contract","internet_service"]
X=df[features]; y=df["churn"]
cat=["contract","internet_service"]; num=["age","tenure_months","monthly_charges","support_calls"]
prep=ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(handle_unknown="ignore"),cat)])
model=Pipeline([("prep",prep),("clf",LogisticRegression(max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
model.fit(Xtr,ytr); pred=model.predict(Xte)
m1,m2,m3,m4=st.columns(4)
m1.metric("Accuracy",f"{accuracy_score(yte,pred):.2%}")
m2.metric("Precision",f"{precision_score(yte,pred):.2%}")
m3.metric("Recall",f"{recall_score(yte,pred):.2%}")
m4.metric("F1 Score",f"{f1_score(yte,pred):.2%}")

df["churn_probability"]=model.predict_proba(X)[:,1]
st.subheader("High-Risk Customers")
st.dataframe(df.sort_values("churn_probability",ascending=False).head(20),use_container_width=True,hide_index=True)
st.info("Business insight: customers with short tenure, frequent support calls, higher monthly charges, and month-to-month contracts are treated as higher-risk segments by the model. Use these segments for targeted retention campaigns.")
