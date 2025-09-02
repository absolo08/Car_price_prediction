from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained pipeline
model = joblib.load("car_price_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect data from form
        Year = int(request.form['Year'])
        Present_Price = float(request.form['Present_Price'])
        Kms_Driven = int(request.form['Kms_Driven'])
        Owner = int(request.form['Owner'])
        Fuel_Type = request.form['Fuel_Type']
        Seller_Type = request.form['Seller_Type']
        Transmission = request.form['Transmission']

        # Create DataFrame with consistent column names
        input_df = pd.DataFrame([{
            'Fuel_Type': Fuel_Type,
            'Seller_Type': Seller_Type,
            'Transmission': Transmission,
            'Year': Year,
            'Present_Price': Present_Price,
            'Kms_Driven': Kms_Driven,
            'Owner': Owner
        }])

        # Predict using pipeline
        pred = model.predict(input_df)
        price_in_lakhs = round(pred[0], 2)

        # Convert Lakhs → full value (Indian dataset stores in Lakhs)
        price_in_naira = int(price_in_lakhs * 100000)

        # Format with commas (e.g., ₦ 525,000)
        formatted_price = "₦ {:,}".format(price_in_naira)

        return render_template("index.html", prediction_text=f"Estimated Price: {formatted_price}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
