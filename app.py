from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Flask app initialization
application = Flask(__name__)
app = application


@app.route('/')
def index():
    return render_template('index.html')  # basic welcome page


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # form page

    else:
        # Get user input
        data = CustomData(
            Gender=request.form.get('Gender'),
            Married=request.form.get('Married'),
            Dependents=request.form.get('Dependents'),
            Education=request.form.get('Education'),
            Self_Employed=request.form.get('Self_Employed'),
            ApplicantIncome=float(request.form.get('ApplicantIncome')),
            CoapplicantIncome=float(request.form.get('CoapplicantIncome')),
            LoanAmount=float(request.form.get('LoanAmount')),
            Loan_Amount_Term=float(request.form.get('Loan_Amount_Term')),
            Credit_History=float(request.form.get('Credit_History')),
            Property_Area=request.form.get('Property_Area')
        )

        # Convert to DataFrame
        final_df = data.get_data_as_dataframe()

        # Predict
        pipeline = PredictPipeline()
        pred = pipeline.predict(final_df)

        # Convert prediction
        result = "Loan Approved " if pred[0] == 1 else "Loan Rejected "

        return render_template('result.html', prediction=result)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

