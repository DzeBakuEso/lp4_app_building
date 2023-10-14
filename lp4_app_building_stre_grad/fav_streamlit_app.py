#below we import our packages 

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

#below we our Load our model and the data processing tools.
# Loading the model
with open('model_xgb_best.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Loading the Scaler
with open('scaler_X.pkl', 'rb') as file:
    loaded_scaler_X = pickle.load(file)

# Loading Feature Names
with open('feature_names.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# Loading Categorical encoder
with open('train_categories.pkl', 'rb') as file:
    categories = pickle.load(file)


#below we process the loaded dataset
def preprocess_data(input_data, categories, feature_names, scaler):
    #below we perform One-hot encoding for categorical variables
    input_data_encoded = pd.get_dummies(input_data, columns=['city', 'state', 'family'])
    
    #here we are ensuring all columns from original model are present in input data, if not, we add them with a default value (0)
    for col in feature_names:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    
    #here we reorder columns according to original model's feature order
    input_data_encoded = input_data_encoded[feature_names]
    
    # below here we call the scaler on the numnerical columns
    input_scaled = scaler.transform(input_data_encoded)
    
    return input_scaled

#below this fumction calls the model on the process data 
def predict_sales(preprocessed_data, model):
    #here we predict sales using the trained model
    forecast = model.predict(preprocessed_data)
    return max(0, forecast[0])

# this code below helps users to navigate the page 
st.sidebar.title('Navigation')
page_selection = st.sidebar.radio('Go to', ['Prediction', 'Model Information'])

if page_selection == 'Prediction':
    st.title('Sales Prediction App')

#below this fucntion is used to accept the inputs from the user
    def get_user_input():

    
        onpromotion = st.number_input('Enter the number of items on promotion', value=0, min_value=0)
        date_input = st.date_input('Select a Date', datetime.now())
        selected_city = st.selectbox('Select City', ['Quito', 'Esmeraldas', 'Machala', 'Libertad', 'Guayaquil', 'Playas', 'Loja',
             'El Carmen', 'Manta', 'Santo Domingo', 'Ambato', 'Guaranda', 'Ibarra',
             'Latacunga', 'Riobamba', 'Quevedo', 'Puyo', 'Daule', 'Salinas', 'Cuenca',
            'Cayambe', 'Babahoyo'])
        selected_state = st.selectbox('Select State', ['Pichincha', 'Esmeraldas', 'El Oro', 'Guayas', 'Loja', 'Manabi',
             'Santo Domingo de los Tsachilas', 'Tungurahua', 'Bolivar', 'Imbabura',
            'Cotopaxi', 'Chimborazo', 'Los Rios', 'Pastaza', 'Santa Elena', 'Azuay'])
        selected_family = st.selectbox('Select Family', ['CELEBRATION', 'BABY CARE', 'HOME AND KITCHEN I', 'DAIRY', 'GROCERY I',
            'PRODUCE', 'HOME CARE', 'FROZEN FOODS', 'AUTOMOTIVE', 'PERSONAL CARE',
            'SEAFOOD', 'LAWN AND GARDEN', 'PET SUPPLIES', 'MAGAZINES', 'PREPARED FOODS',
            'LIQUOR,WINE,BEER', 'BOOKS', 'HOME APPLIANCES', 'EGGS', 'CLEANING', 'POULTRY',
            'LADIESWEAR', 'MEATS', 'DELI', 'BEAUTY', 'HOME AND KITCHEN II',
            'SCHOOL AND OFFICE SUPPLIES', 'BEVERAGES', 'PLAYERS AND ELECTRONICS',
            'HARDWARE', 'GROCERY II', 'LINGERIE', 'BREAD/BAKERY'])

        day_of_week = date_input.weekday()
        data = {
            'onpromotion': [onpromotion],
            'day_of_week': [day_of_week],
            'lag_1': [0], 
            'rolling_mean': [0], 
            'city': [selected_city], 
            'state': [selected_state], 
            'family': [selected_family]
        }
        return pd.DataFrame(data)

    user_input = get_user_input()


    preprocessed_data = preprocess_data(user_input, categories, feature_names, loaded_scaler_X)
    
    if st.button('Predict Sales'):
            with st.spinner('Predicting...'):
                try:
                    prediction = predict_sales(preprocessed_data, loaded_model)
                    st.success(f"Predicted Sales: ${prediction:.2f}")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    st.markdown('**Note:** This is a demo and the prediction might not be accurate.')
    
    # below here we create a feedback mechanism to accept users feedback
    st.markdown('### Feedback')
    feedback_options = st.radio('Was the prediction accurate?', ['👍 Yes', '👎 No'])
    feedback_text = st.text_area("Any additional feedback or suggestions?")
    if st.button('Submit Feedback'):
        # Here we can store the feedback 
            st.success('Thank you for your feedback!')

elif page_selection == 'Model Information':
    st.title("About the Model")
    
    st.subheader("Model Accuracy")
    st.write("The best model, equipped with its fine-tuned parameters, achieved an impressive Root Mean Squared Logarithmic Error (RMSLE) of 0.0054. This low RMSLE score underscores the model's accuracy.")
    
    st.subheader("Training Data")
    st.markdown("""
        ### Data Sampling & Training

        The model was trained on a systematically sampled subset of the dataset, where every 30th data point was selected from a total of 3,000,000 records, resulting in 100,000 data points. This subset spans from 2013 to 2017, with a final size of 10,000 records that were used for training.

        #### Pros of this approach:

        - **Reduction in Computational Cost**: Training on a systematically sampled subset is computationally less intensive than using the entire dataset. This can result in faster training times and reduced resource consumption.
        - **Uniform Distribution**: Systematic sampling ensures a uniform distribution across the dataset, potentially capturing the overall trend and variance in the data.
        - **Ease of Implementation**: Systematic sampling is straightforward to implement and doesn't require complex algorithms or randomness.
        """)

    st.subheader("Last Updated")
    st.write("The model was last updated on 10th October 2023.")
    
    st.subheader("Model Architecture")
    st.write("We used a XGBoost Model with the following configurations:")
    st.write("- Learning Rate: 0.3")
    st.write("- Maximum Depth: 6")
    st.write("- Number of Trees (Estimators): 200")
