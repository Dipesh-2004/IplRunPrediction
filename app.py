from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained pipeline safely
model = joblib.load("pipe.joblib")

# Mapping team names to cities to determine home_advantage automatically
city_mapping = {
    'Sunrisers Hyderabad': 'Hyderabad',
    'Mumbai Indians': 'Mumbai',
    'Royal Challengers Bangalore': 'Bangalore',
    'Kolkata Knight Riders': 'Kolkata',
    'Kings XI Punjab': 'Punjab',
    'Chennai Super Kings': 'Chennai',
    'Rajasthan Royals': 'Rajasthan',
    'Delhi Capitals': 'Delhi'
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Capture user fields safely from the HTML form elements
        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]
        city = request.form["city"]
        target = int(request.form["target"])
        runs_left = int(request.form["runs_left"])
        balls_left = int(request.form["balls_left"])
        wickets_left = int(request.form["wickets_left"])
        
        # 2. Math calculations to match exactly what your test.py expects
        balls_bowled = 120 - balls_left
        runs_scored = (target - 1) - runs_left
        
        # Guard against zero-division errors safely
        crr = float(runs_scored / (balls_bowled / 6)) if balls_bowled > 0 else 0.0
        rrr = float(runs_left / (balls_left / 6)) if balls_left > 0 else 0.0
        
        # Determine home advantage
        home_city = city_mapping.get(batting_team, "")
        home_advantage = 1 if home_city.lower() == city.lower() else 0

        # 3. Put data inside a dictionary with strict types
        data = {
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "city": city,
            "runs_left": runs_left,
            "balls_left": balls_left,
            "wickets_left": wickets_left,
            "target": target,
            "crr": crr,
            "rrr": rrr,
            "home_advantage": home_advantage
        }

        # Convert to Pandas DataFrame
        input_df = pd.DataFrame([data])

        # ⚠️ THE BULLETPROOF FIX:
        # Instead of manual layout matching, force the DataFrame to copy 
        # the exact order stored inside the trained model metadata properties.
        if hasattr(model, 'feature_names_in_'):
            input_df = input_df[list(model.feature_names_in_)]

        # 4. Run prediction probabilities
        probabilities = model.predict_proba(input_df)[0]
        batting_team_prob = round(probabilities[1] * 100, 1)
        bowling_team_prob = round(probabilities[0] * 100, 1)
        
        result_text = f"{batting_team} Will Win 🏆" if batting_team_prob >= 50.0 else f"{batting_team} Will Lose ❌"
        
        return render_template(
            "index.html",
            prediction=[batting_team_prob, bowling_team_prob],
            result=result_text,
            batting_team=batting_team,
            bowling_team=bowling_team
        )
        
    except Exception as e:
        print(f"Server Execution Error: {e}")
        return f"Prediction Failed: {e}", 400

if __name__ == "__main__":
    app.run(debug=True)