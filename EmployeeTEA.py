import streamlit as st
import numpy as np
import pandas as pd
import joblib
from joblib import load
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator, TransformerMixin

# ---Setting Streamlit Page Configuration---
st.set_page_config(
    page_title="Employee Training Effectiveness Analysis Dashboard",page_icon="📊",layout="wide")

# --- Loading Dataset by Database---
@st.cache_data
def load_data():
    conn = sqlite3.connect('ETEA.db')
    df = pd.read_sql_query("SELECT * FROM ETEA", conn)
    conn.close()
    return df

df=load_data()

# Main content
st.header("Employee Training Effectiveness Analyzer")
st.sidebar.markdown(""" 
        Welcome to the Employee Training Effectiveness Analysis Dashboard!  
        Here you can:
        - Explore the Cleaned Dataset
        - See Powerful Visual Analysis
        - Make Predictions
        
        Use this sidebar to navigate! """)

# --- KPIs ---
st.markdown("## 📌Metrics")
col1, = st.columns(1)
total_hours = df["Trained Hours"].sum()
col1.metric("Total Trained Hours", f"{total_hours:,.0f} hours")

# --- Sidebar Navigation---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", ["🏠Dashboard Home", "📄 Dataset Overview", "📈 Analysis", "🔮 Predict New Data"])

# --- Pages ---
if selection == "🏠Dashboard Home":
    department_types = ["HR", "Productions", "Sales", "Marketing", "Finance", "IT", "Operations"]
    df_filtered = df[df['Department'].isin(department_types)]
    department_counts = df_filtered['Department'].value_counts()
    departments = department_counts.index.tolist()
    counts = department_counts.values.tolist()
    fig11 = go.Figure(data=[
        go.Bar(x=departments,y=counts,text=counts,textposition='outside',marker_color=['skyblue', 'lightcoral', 'lightgreen', 'lightsalmon', 'green', 'blue', 'orange'])])
    fig11.update_layout(
        title='Total Employees in Each Department',xaxis_title='Department',yaxis_title='Number of Employees',xaxis_tickangle=-45,plot_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=40))
    st.plotly_chart(fig11)

    fig22 = px.scatter(df,x='Age',y='Work Experience',title='Age and Work Experience',color_discrete_sequence=['gray'],)
    fig22.update_traces(marker=dict(line=dict(width=1, color='green')))
    fig22.update_layout(xaxis_title='Age',yaxis_title='Work Experience',plot_bgcolor='white',margin=dict(l=40, r=20, t=40, b=40),showlegend=False)
    st.plotly_chart(fig22)

    training_program_counts = df['Training Program'].value_counts()
    programs = training_program_counts.index.tolist()
    counts = training_program_counts.values.tolist()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=programs,y=counts,mode='lines+markers+text',text=counts,textposition='top center',line=dict(color='royalblue', width=3),marker=dict(size=8)))
    fig3.update_layout(title='Training Program Trend',xaxis_title='Types of Program',yaxis_title='Trained Employees in Quantity',xaxis_tickangle=-45,plot_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=40),showlegend=False)
    st.plotly_chart(fig3)

    #Aggregate total trained hours per program
    total_tPro_hours = df.groupby("Training Program")["Trained Hours"].sum().sort_values(ascending=False)
    programs = total_tPro_hours.index.tolist()
    hours = total_tPro_hours.values.tolist()
    # Line Chart Alternative
    fig4_line = go.Figure()
    fig4_line.add_trace(go.Scatter(x=programs,y=hours,mode='lines+markers+text',text=hours,textposition='top center',line=dict(color='green', width=3),marker=dict(size=8)))
    fig4_line.update_layout(title='Trained Hours Per Training Program (Line Chart)',xaxis_title='Training Program',yaxis_title='Trained Hours',xaxis_tickangle=-45,
        plot_bgcolor='white',margin=dict(l=40, r=20, t=40, b=40),showlegend=False)
    st.plotly_chart(fig4_line)

elif selection == "📄 Dataset Overview":
    st.title("Dataset Overview")
    st.dataframe(df)
    st.write("Summary Statistics:")
    st.write(df.describe())
    # Histogram
    st.subheader("Distribution of Post Training Score")
    fig2 = px.histogram(df, x="Post Training Score") 
    st.plotly_chart(fig2)

    #Scatter Plot
    st.subheader("Relationship By Columns")
    x_feature = st.selectbox("Select a Column", df.columns, key="x_feature_selectbox")
    y_feature = st.selectbox("Select a Column", df.columns, key="y_feature_selectbox")
    color_col= st.selectbox("Color by", df.columns)
    fig3 = px.scatter(df, x=x_feature, y=y_feature, color=color_col)
    st.plotly_chart(fig3)

    st.subheader("Correlation Heatmap")
    # Correlation matrix
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    st.corr = numeric_df.corr()
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8,4))
    sns.heatmap(st.corr, annot=True, cmap = plt.cm.Blues, fmt=".2f")  
    st.pyplot(fig)

elif selection == "📈 Analysis":
    st.markdown("Lets Analyze Our Dataset!")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Basic Statistics")
        st.write(df.describe())
    
    with col2:
        st.subheader("Data Visualization") 
        plot_type = st.selectbox("Plot Type",["Histogram", "Scatter Plot", "Box Plot"])
        
        if plot_type == "Histogram":
            column = st.selectbox("Select column", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True, ax=ax)
            st.pyplot(fig)
            
        elif plot_type == "Scatter Plot":
            x_axis = st.selectbox("X Axis", df.columns)
            y_axis = st.selectbox("Y Axis", df.columns)
            color_by = st.selectbox("Color By", [None] + list(df.select_dtypes(include=['object', 'category']).columns))     
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by, hover_data=df.columns)
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Box Plot":
            column = st.selectbox("Select column", df.columns)
            group_by = st.selectbox("Group By", [None] + list(df.select_dtypes(include=['object', 'category']).columns))  
            fig, ax = plt.subplots()
            if group_by:
                sns.boxplot(data=df, x=group_by, y=column, ax=ax)
            else:
                sns.boxplot(data=df, y=column, ax=ax)
            st.pyplot(fig)

        else: # Correlation matrix
            numeric_df = df.select_dtypes(include=['float64', 'int64'])
            corr_matrix = numeric_df.corr()
            # Plot heatmap
            fig, ax = plt.subplots(figsize=(8,4))
            sns.heatmap(corr_matrix, annot=True, cmap=plt.cm.Blues, fmt=".2f")
            st.pyplot(fig)

elif selection == "🔮 Predict New Data":
    st.title("Predict New Data")
    st.subheader("Input Features")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    with col1:
        education = st.selectbox("Education",["High School", "Bachelor's", "Master's", "PhD" ])

    with col2:
        work_experience = st.slider("Work Experience", 18, 60, 30)

    with col3:
        training_program = st.selectbox("Training program", ['Compliance','Customer Service','Leadership','Soft Skills','Technical Skills'])
        
    with col4:
        training_type = st.selectbox("Training Type",['Online', 'Hybrid', 'In-Person'])

    with col5:
        pre_training = st.number_input("Pre Training Score",0.00, 100.00, 10.00)  
    
    with col6:
        post_training = st.number_input("Post Training Score",0.00, 100.00, 10.00)
    
    with col7:
        peer_learning = st.selectbox("Peer Learning", ["Yes", "No"])

    with col8:
        feedback = st.slider("Feedback Score", 1, 10, 5)

    education_mapping = {'PhD':4, "Master's":3, "Bachelor's":2, 'High School':1}
    education_encoded =education_mapping[education]

    training_mapping = {'Compliance':1, 'Customer Service':2, 'Leadership':3,'Soft Skills':4, 'Technical Skills':5}
    training_encoded =training_mapping[training_program]

    traintype_mapping ={'Online':1, 'Hybrid':2, 'In-Person':3}
    traintype_encoded =traintype_mapping[training_type]

    peer_mapping ={'No':0, 'Yes':1}
    peer_encoded =peer_mapping[peer_learning]

    input_data = np.array([[education_encoded, work_experience, training_encoded, traintype_encoded, pre_training, post_training, peer_encoded, feedback]])

    # --- Model Requirements---
    class PerformancePredictor(BaseEstimator, TransformerMixin):
        def __init__(self, model):
            self.model = model
        
        def fit(self, X, y=None):
            # Ensure feature names are strings
            if isinstance(X, pd.DataFrame):
                X.columns = X.columns.map(str)
                self.feature_names_in_ = X.columns
            else:
                self.feature_names_in_ = None
            return self
        
        def transform(self, X):
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            
            # Ensure column names are strings
            if isinstance(X, pd.DataFrame):
                X.columns = X.columns.map(str)
            
            pred_perf = self.model.predict(X)
            X_new = X.copy()
            X_new['predicted_performance'] = pred_perf
            return X_new

    # --- Loading Model---
    @st.cache_resource
    def load_model():
        return joblib.load("mlmodel_mlp_rfc.pkl")

    model_ = joblib.load("mlmodel_mlp_rfc.pkl")
    mlp = model_["regressor"]
    rfc = model_["clf_pipeline"]

    if st.button("🔮Predict"):
        #Make prediction
        performance_imp = mlp.predict(input_data)[0]
        st.success(f"🌟Performance Score: {performance_imp:.2f}")
        
        promotion_ = rfc.predict(input_data)[0]
        yesno= "Yes" if promotion_ == 1 else "No"
        st.success(f"✨Promotion Eligibility: {yesno}")
