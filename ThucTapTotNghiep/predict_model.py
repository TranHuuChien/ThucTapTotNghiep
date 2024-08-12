import numpy as np
import pandas as pd
import pickle
import sklearn
from flask import Flask, render_template, request

app = Flask(__name__)

model = pickle.load(open('./ThucTapTotNghiep/train_model.pkl', 'rb'))
preprocessor = pickle.load(open('./ThucTapTotNghiep/preprocessor.pkl', 'rb'))

@app.route('/')
def Home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Gathering inputs
    account_length = int(request.form.get('account_length'))
    international_plan = request.form.get('international_plan')
    voice_mail_plan = request.form.get('voice_mail_plan')
    number_vmail_messages = int(request.form.get('number_vmail_messages'))

    total_day_minutes = float(request.form.get('total_day_minutes'))
    total_day_calls = int(request.form.get('total_day_calls'))

    total_eve_minutes = float(request.form.get('total_eve_minutes'))
    total_eve_calls = int(request.form.get('total_eve_calls'))

    total_night_minutes = float(request.form.get('total_night_minutes'))
    total_night_calls = int(request.form.get('total_night_calls'))

    total_intl_minutes = float(request.form.get('total_intl_minutes'))
    total_intl_calls = int(request.form.get('total_intl_calls'))

    number_customer_service_calls = int(request.form.get('number_customer_service_calls'))

    inputs = inputs = pd.DataFrame(np.array([account_length, international_plan, voice_mail_plan, number_vmail_messages, total_day_minutes
        , total_day_calls, total_eve_minutes, total_eve_calls,total_night_minutes, total_night_calls,
        total_intl_minutes, total_intl_calls, number_customer_service_calls]).reshape(1, -1), 
                                    columns=['account_length', 'international_plan', 'voice_mail_plan', 'number_vmail_messages',
                                    'total_day_minutes', 'total_day_calls', 'total_eve_minutes','total_eve_calls','total_night_minutes'
                                    ,'total_night_calls','total_intl_minutes','total_intl_calls','number_customer_service_calls'])

    input_processed = preprocessor.transform(inputs)

    prediction = model.predict(input_processed)

    # Generate churn risk scores
    churn_risk_scores = np.round(model.predict_proba(input_processed)[:, 1] * 100,2)

    # Churn flag
    if prediction == 1:
        prediction = 'YES'
    else:
        prediction = 'NO'

    return render_template('predict.html', prediction=prediction, churn_risk_scores=churn_risk_scores, inputs=request.form)

if __name__ == '__main__':
    app.run(debug=True)
