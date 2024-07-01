import streamlit as st
from src.pipeline.prediction_pipeline import GetFeature, Prediction


st.header("Lung Cancer Prediction Center")
age = st.number_input("please enter age",value=0,max_value=100)
gender = st.selectbox("Gender", ["M", "F"])
smoking = st.selectbox("Smokes", ["yes", "no"])
yellow_fingers = st.selectbox("Have yellow finger", ["yes", "no"])
anxiety = st.selectbox("have anxiety", ["yes", "no"])
peer_pressure = st.selectbox("have peer pressure", ["yes", "no"])
chronic_disease = st.selectbox("have chronic disease", ["yes", "no"])
fatigue = st.selectbox("have fatigue", ["yes", "no"])
allergy = st.selectbox("have allergy", ["yes", "no"])
wheezing = st.selectbox("wheezing", ["yes", "no"])
alcohol_consuming = st.selectbox("alcohol consumption", ["yes", "no"])
coughing = st.selectbox("cough", ["yes", "no"])
shortness_of_breath = st.selectbox("shortness of breath", ["yes", "no"])
swallowing_difficulty = st.selectbox("swallowing difficulty", ["yes", "no"])
chest_pain = st.selectbox("chest pain", ["yes", "no"])

ok = st.button("PREDICT")

if ok:
    features = GetFeature(gender, age, smoking, yellow_fingers, anxiety,
                          peer_pressure, chronic_disease, fatigue, allergy,
                          wheezing,
                          alcohol_consuming, coughing,
                          shortness_of_breath,
                          swallowing_difficulty, chest_pain)


    feature_df = features.to_dataframe()
    print(feature_df)
    pred = Prediction()
    op = pred.predict(feature_df)
    o=int(op)
    if o == 1:
        st.subheader("Cancer detected")
