from flask import Flask, request, render_template, redirect, url_for, session
import joblib
import numpy as np
import json
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Secret key for session management

# Load the trained model
model = joblib.load('Extra_model.pkl')

# File to store user data
USER_FILE = 'users.json'

# Load users from the file
def load_users():
    try:
        with open(USER_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Save users to the file
def save_users(users):
    with open(USER_FILE, 'w') as f:
        json.dump(users, f)

@app.route('/')
def home():
    if 'username' in session:
        return redirect(url_for('predict'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    users = load_users()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users:
            return "User already exists! Please log in."
        
        # Store hashed password
        users[username] = generate_password_hash(password)
        save_users(users)  # Save to file
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    users = load_users()
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('predict'))
        else:
            return "Invalid username or password."

    return render_template('login.html')



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Get input data from the form
            ID = int(request.form['ID'])
            Timestamp = int(request.form['Timestamp'])
            ACC_x = int(request.form['ACC_x'])
            ACC_y = int(request.form['ACC_y'])
            ACC_z = float(request.form['ACC_z'])
            BVP = int(request.form['BVP'])
            EDA = int(request.form['EDA'])
            HR = int(request.form['HR'])
            IBI_d = int(request.form['IBI_d'])
            TEMP = float(request.form['TEMP'])  # Ensure TEMP can handle decimals
            
            # Create input array for prediction
            input_data = np.array([[ID, Timestamp, ACC_x, ACC_y, ACC_z, BVP, EDA, HR, IBI_d, TEMP]])

            # Make prediction
            prediction = model.predict(input_data)

            # Define result messages
            if prediction == 1:
                result = f"Your temperature is {TEMP}°C, indicating that you are in stress."
            else:
                result = f"Your temperature is {TEMP}°C, and you are not in stress."

            # Return prediction result
            return render_template('predict.html', prediction_text=result)

        except Exception as e:
            return str(e)

    return render_template('predict.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
