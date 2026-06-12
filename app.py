import streamlit as st
import pandas as pd
import joblib
import os

#CONFIG MUST BE FIRST
st.set_page_config(page_title="AI Project Predictor", page_icon="🚀")

st.title(" AI Team Success Predictor")

# FILE LOADING
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "project_success_brain.pkl")

try:
    model = joblib.load(model_path)
    st.success(f"Brain loaded successfully from: {current_dir}") # Debugging
except FileNotFoundError:
    st.error("Model Not Found!")
    st.code(f"Looking for file at:\n{model_path}")
    st.warning("Make sure 'project_success_brain.pkl' is in the SAME folder as 'app.py'")
    st.stop()

# 3. INPUTS 
st.sidebar.header("Configuration")
project_type = st.sidebar.selectbox("Project Type", ("AI_Model", "Web_App", "Mobile_App", "Research"))
team_size = st.sidebar.slider("Team Size", 3, 10, 5)

max_val = team_size * 10
coding = st.sidebar.slider("Total Coding Skill", 0, max_val, int(max_val * 0.5))
mgmt = st.sidebar.slider("Total Management Skill", 0, max_val, int(max_val * 0.3))
design = st.sidebar.slider("Total Design Skill", 0, max_val, int(max_val * 0.3))
math = st.sidebar.slider("Total Math Skill", 0, max_val, int(max_val * 0.3))

# 4. PREDICT 
if st.button("Predict Success"):
    p_map = {"AI_Model": 0, "Web_App": 1, "Mobile_App": 2, "Research": 3}
    input_data = pd.DataFrame([[p_map[project_type], team_size, coding, mgmt, design, math]], 
                              columns=['Project_Type', 'Team_Size', 'Skill_Coding', 'Skill_Mgmt', 'Skill_Design', 'Skill_Math'])
    
    prediction = model.predict(input_data)[0] * 100
    
    st.divider()
    st.metric("Probability", f"{prediction:.2f}%")
    st.progress(int(prediction))
