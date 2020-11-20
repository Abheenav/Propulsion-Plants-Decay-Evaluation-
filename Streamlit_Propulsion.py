# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:26:43 2020

@author: Abheenav
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from PIL import Image



pickle_in = open("finalized_model_GT_C_D1.sav","rb")
modelc=pickle.load(pickle_in)
pickle_in = open("finalized_model_GT_T_D1.sav","rb")
modelt=pickle.load(pickle_in)


def welcome():
    return "Welcome All"


def predict_propulsion_c(lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow):
    
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: lever_position
        in: query
        type: float
        required: true
      - name: ship_speed
        in: query
        type: float
        required: true
      - name: gt_shaft
        in: query
        type: float
        required: true
      - name: gt_rate
        in: query
        type: float
        required: true
      - name: gg_rate
        in: query
        type: float
        required: true
      - name: sp_torque
        in: query
        type: float
        required: true
      - name: pp_torque
        in: query
        type: float
        required: true
      - name: hpt_temp
        in: query
        type: float
        required: true
      - name: gt_c_o_temp
        in: query
        type: float
        required: true
      - name: hpt_pressure
        in: query
        type: float
        required: true
      - name: gt_c_i_pressure
        in: query
        type: float
        required: true
      - name: gt_c_o_pressure
        in: query
        type: float
        required: true
      - name: gt_exhaust_pressure
        in: query
        type: float
        required: true
      - name: turbine_inj_control
        in: query
        type: float
        required: true
      - name: fuel_flow
        in: query
        type: float
        required: true
        
    responses:
        200:
            description: The output values
        
    """
   
    
    prediction=modelc.predict(([[lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow]]))
    print(prediction)
    return prediction
def predict_propulsion_t(lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow):
    
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: lever_position
        in: query
        type: float
        required: true
      - name: ship_speed
        in: query
        type: float
        required: true
      - name: gt_shaft
        in: query
        type: float
        required: true
      - name: gt_rate
        in: query
        type: float
        required: true
      - name: gg_rate
        in: query
        type: float
        required: true
      - name: sp_torque
        in: query
        type: float
        required: true
      - name: pp_torque
        in: query
        type: float
        required: true
      - name: hpt_temp
        in: query
        type: float
        required: true
      - name: gt_c_o_temp
        in: query
        type: float
        required: true
      - name: hpt_pressure
        in: query
        type: float
        required: true
      - name: gt_c_i_pressure
        in: query
        type: float
        required: true
      - name: gt_c_o_pressure
        in: query
        type: float
        required: true
      - name: gt_exhaust_pressure
        in: query
        type: float
        required: true
      - name: turbine_inj_control
        in: query
        type: float
        required: true
      - name: fuel_flow
        in: query
        type: float
        required: true
        
    responses:
        200:
            description: The output values
        
    """
   
    #prediction=classifier.predict(tmatrix)
    #ab = np.array([[lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow]])
    #abt = ab.T
    #prediction=classifier.predict(abt)
    prediction=modelt.predict(([[lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow]]))
    print(prediction)
    return prediction



def main():
    st.title("Propulsion")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Propulsion Plants Decay Evaluation( ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    lever_position=st.text_input("lever_position")
    ship_speed=st.text_input("ship_speed")
    gt_shaft=st.text_input("gt_shaft")
    gt_rate=st.text_input("gt_rate")
    gg_rate=st.text_input("gg_rate")
    sp_torque=st.text_input("sp_torque")
    pp_torque=st.text_input("pp_torque")
    hpt_temp=st.text_input("hpt_temp")
    gt_c_o_temp=st.text_input("gt_c_o_temp")
    hpt_pressure=st.text_input("hpt_pressure")
    gt_c_i_pressure=st.text_input("gt_c_i_pressure")
    gt_c_o_pressure=st.text_input("gt_c_o_pressure")
    gt_exhaust_pressure=st.text_input("gt_exhaust_pressure")
    turbine_inj_control=st.text_input("turbine_inj_control")
    fuel_flow=st.text_input("fuel_flow")
    #matrix = np.array([[lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow]])
    #tmatrix = matrix.T
    resultc=""
    
    if st.button("Predict GT Compressor decay"):
        #result=predict_propulsion(tmatrix)
        resultc=predict_propulsion_c(lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow)
        st.success('The output of GT Compressor decay is {}'.format(resultc))
    resultt=""
    if st.button("Predict GT Turbine decay"):
        resultt=predict_propulsion_t(lever_position, ship_speed, gt_shaft, gt_rate, gg_rate, sp_torque, pp_torque, hpt_temp, gt_c_o_temp, hpt_pressure, gt_c_i_pressure, gt_c_o_pressure, gt_exhaust_pressure, turbine_inj_control, fuel_flow)
        st.success('The output of GT Turbine decay is {}'.format(resultt))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
