import gradio as gr
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load model
with open('churn_model.pkl', 'rb') as file:
    rfc_loaded = pickle.load(file)

# Load preprocessor
with open('churn_pipeline.pkl', 'rb') as file:
    preprocessor_loaded = pickle.load(file)

# Creating the preprocessing pipeline
categorical_features = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']

numerical_features = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges']

categorical_transformer = OneHotEncoder(drop='first')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('num', numerical_transformer, numerical_features)
    ]
)

def predict(SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
            MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
            StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure):

    data = [[SeniorCitizen, MonthlyCharges, TotalCharges, gender, Partner, Dependents, PhoneService, 
             MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
             StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, tenure]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    processed_data = preprocessor_loaded.transform(new_df)
    single = rfc_loaded.predict(processed_data)

    if single == 1:
        return "This customer is likely to be churned."
    else:
        return "This customer is likely to continue."

input_components = []
# Define tooltips as a dictionary

with gr.Blocks(title="Customer Churn Prediction", theme= "monochrome") as iface:
     
    with gr.Row():
        title = gr.Label("Customer Churn Prediction")
        
    with gr.Row():
        gr.Markdown(" # Predict whether a customer is likely to churn.")

    with gr.Row():
        
        # Define the input components
            input_components = [
                    gr.Dropdown(label='SeniorCitizen', choices=[0, 1]),
                    gr.Number(label='MonthlyCharges'),
                    gr.Number(label='TotalCharges'),
                    gr.Dropdown(label='gender', choices=["Male", "Female"]),
                    gr.Dropdown(label='Partner', choices=["Yes", "No"]),
                    gr.Dropdown(label='Dependents', choices=["Yes", "No"]),
                    gr.Dropdown(label='PhoneService', choices=["Yes", "No"]),
                    gr.Dropdown(label='MultipleLines', choices=["Yes", "No", "No phone service"]),
                    gr.Dropdown(label='InternetService', choices=["DSL", "Fiber optic", "No"]),
                    gr.Dropdown(label='OnlineSecurity', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='OnlineBackup', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='DeviceProtection', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='TechSupport', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='StreamingTV', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='StreamingMovies', choices=["Yes", "No", "No internet service"]),
                    gr.Dropdown(label='Contract', choices=["Month-to-Month", "One year", "Two year"]),
                    gr.Dropdown(label='PaperlessBilling', choices=["Yes", "No"]),
                    gr.Dropdown(label='PaymentMethod', choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
                    gr.Dropdown(label='tenure', choices=["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
                ]


    with gr.Row():
        pred = gr.Button('Predict')
        
    
    
    output_components = gr.Label(label='Churn') 
    
    pred.click(fn=predict,
            inputs=input_components,
            outputs=output_components,
            )        



iface.launch(share=True)
