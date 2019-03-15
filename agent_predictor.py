
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os 

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/bulkload",methods=['GET','POST'])
def bulk_load():
    if request.method=='POST':
        file = request.files['customers']
        
        if not file: 
            return render_template('index.html', label="No file")
        
        df=pd.read_csv(file)
        
        #TODO
        #TODO
        #TODO
        
    return render_template('bulkload.html',lable=lablel)

@app.route('/predict', methods=['POST'])
def make_prediction():
    
    
    if request.method=='POST':
        
        national_producer_number = request.form.get('national_producer_number')
        gender = 1 if request.form.get('gender') =='Yes' else 0
        age = request.form.get('age')
        salary = float(request.form.get('salary'))
        los = float(request.form.get('los'))
        maratial_status = 1 if request.form.get('maratial_status')=='Yes' else 0
        mailing_state =request.form.get('mailing_state')
        resident = 1 if request.form.get('resident')=='Yes' else 0
        crop = 1 if request.form.get('crop') =='Yes' else 0
        surety = 1 if request.form.get('surety') =='Yes' else 0
        accident_and_health = 1 if request.form.get('accident_and_health') =='Yes' else 0
        life = 1 if request.form.get('life') =='Yes' else 0
        variable_life_variable_annuity = 1 if request.form.get('variable_life_variable_annuity') =='Yes' else 0
        personal_lines = 1 if request.form.get('personal_lines') =='Yes' else 0
        credit = 1 if request.form.get('credit') =='Yes' else 0
        excess_and_surplus_lines = 1 if request.form.get('excess_and_surplus_lines') =='Yes' else 0
        property1 = 1 if request.form.get('property1') =='Yes' else 0
        casualty = 1 if request.form.get('casualty') =='Yes' else 0
        resiprocal_authority = 1 if request.form.get('resiprocal_authority') =='Yes' else 0
        
        
        test_data=pd.DataFrame({'gender':[gender],'age':[age],
                                'salary':[salary], 'los':[los],'maratial_status':[maratial_status],
                                'mailing_state':[mailing_state],'resident':[resident],'crop':[crop],'surety':[surety],
                                'accident_and_health':[accident_and_health],'life':[life],
                                'variable_life_variable_annuity':[variable_life_variable_annuity],
                                'personal_lines':[personal_lines],'credit':[credit],
                                'excess_and_surplus_lines':[excess_and_surplus_lines],
                                'property1':[property1],'casualty':[casualty],
                                'resiprocal_authority':[resiprocal_authority]})
        
        
        test_data_array=test_data.values

        test_data_array[:,5]=labelencoder.transform(test_data_array[:,5])
        
        test_data_onehot_encoded=onehotenc.transform(test_data_array)
        
        test_data_scaled=stdscalar.transform(test_data_onehot_encoded)
                
        '''input_val=np.array([[national_producer_number, gender,age,salary, los,maratial_status,mailing_state,
                            resident,crop,surety,accident_and_health,life,variable_life_variable_annuity,personal_lines,
                            credit,excess_and_surplus_lines,property1,casualty,resiprocal_authority]])'''
        
        prediction=model.predict(test_data_scaled)
        print(prediction)
    
    
        return render_template('index.html', label=prediction)
        
    
if __name__ == "__main__":
    labelencoder_pickle_file=os.path.join(os.pardir,'models','agent_predictor_label_encoder.pkl')
    onehot_pickle_file=os.path.join(os.pardir,'models','agent_predictor_onehot_encoder.pkl')
    scalar_pickle_file=os.path.join(os.pardir,'models','agent_predictor_standardscalar_encoder.pkl')
    model_pickle_file=os.path.join(os.pardir,'models','agent_predictor_model_xgboost.pkl')
    
    with open(labelencoder_pickle_file, 'rb') as handle:
        labelencoder = pickle.load(handle)
        
    with open(onehot_pickle_file,'rb') as handle:
        onehotenc=pickle.load(handle)
        
    with open(scalar_pickle_file,'rb') as handle:
        stdscalar=pickle.load(handle)
        
    with open(model_pickle_file,'rb') as handle:
        model=pickle.load(handle)
        
    app.run(debug=True)
