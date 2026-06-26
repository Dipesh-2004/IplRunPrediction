import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# =========================================================
# 1️⃣ LOAD DATA
# =========================================================
matches = pd.read_csv('matches.csv')
deliveries = pd.read_csv('deliveries.csv')
print("✅ Data Loaded Successfully")

# =========================================================
# 2️⃣ FEATURE ENGINEERING
# =========================================================

# First innings total → Target
total_score_df = deliveries.groupby(['match_id', 'inning']).sum()['total_runs'].reset_index()
total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df['target'] = total_score_df['total_runs'] + 1

match_df = matches.merge(
    total_score_df[['match_id', 'target']],
    left_on='id',
    right_on='match_id'
)

# -----------------------------
# Clean Team Names
# -----------------------------
teams = [
    'Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
    'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
    'Rajasthan Royals', 'Delhi Capitals'
]

for col in ['team1', 'team2', 'winner']:
    match_df[col] = match_df[col].str.replace('Delhi Daredevils', 'Delhi Capitals')

# Filter teams + remove D/L matches
match_df = match_df[
    (match_df['team1'].isin(teams)) &
    (match_df['team2'].isin(teams)) &
    (match_df['dl_applied'] == 0)
]

match_df = match_df[['match_id','city','winner','target']]

# =========================================================
# Merge with deliveries → 2nd innings only
# =========================================================
delivery_df = match_df.merge(deliveries, on='match_id')
delivery_df = delivery_df[delivery_df['inning'] == 2]

runs_col = 'total_runs_y' if 'total_runs_y' in delivery_df.columns else 'total_runs'

# Current score
delivery_df['current_score'] = delivery_df.groupby('match_id')[runs_col].cumsum()

# Runs left
delivery_df['runs_left'] = np.maximum(
    delivery_df['target'] - delivery_df['current_score'], 0
)

# Balls left
delivery_df['balls_left'] = 126 - (
    delivery_df['over']*6 + delivery_df['ball']
)

# Wickets left
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['is_wicket'] = delivery_df['player_dismissed'].apply(
    lambda x: 0 if x == "0" else 1
)

wickets = delivery_df.groupby('match_id')['is_wicket'].cumsum().values
delivery_df['wickets_left'] = 10 - wickets

# Run rates
delivery_df['crr'] = (
    delivery_df['current_score'] * 6
) / (120 - delivery_df['balls_left'])

delivery_df['rrr'] = (
    delivery_df['runs_left'] * 6
) / delivery_df['balls_left']

# Result column
delivery_df['result'] = (
    delivery_df['batting_team'] == delivery_df['winner']
).astype(int)

# =========================================================
# FINAL DATAFRAME
# =========================================================
final_df = delivery_df[[
    'batting_team','bowling_team','city',
    'runs_left','balls_left','wickets_left',
    'target','crr','rrr','result'
]]

final_df = final_df.dropna()
final_df = final_df[final_df['balls_left'] != 0]

# =========================================================
# 3️⃣ ADD HOME GROUND ADVANTAGE
# =========================================================

home_grounds = {
    'Sunrisers Hyderabad': 'Hyderabad',
    'Mumbai Indians': 'Mumbai',
    'Royal Challengers Bangalore': 'Bangalore',
    'Kolkata Knight Riders': 'Kolkata',
    'Kings XI Punjab': 'Punjab',
    'Chennai Super Kings': 'Chennai',
    'Rajasthan Royals': 'Rajasthan',
    'Delhi Capitals': 'Delhi'
}

def home_advantage(row):
    team = row['batting_team']
    city = row['city']
    
    if team in home_grounds:
        return 1 if home_grounds[team].lower() in str(city).lower() else 0
    return 0

final_df['home_advantage'] = final_df.apply(home_advantage, axis=1)

print("✅ Home Advantage Feature Added")

# =========================================================
# 4️⃣ MODEL TRAINING
# =========================================================

X = final_df[[
    'batting_team','bowling_team','city',
    'runs_left','balls_left','wickets_left',
    'target','crr','rrr','home_advantage'
]]

y = final_df['result']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Column Transformer
trf = ColumnTransformer([
    ('cat',
     OneHotEncoder(
         sparse_output=False,
         drop='first',
         handle_unknown='ignore'
     ),
     ['batting_team','bowling_team','city'])
], remainder='passthrough')

# Pipeline
pipe = Pipeline([
    ('preprocess', trf),
    ('model', LogisticRegression(solver='liblinear'))
])

print("⏳ Training model...")
pipe.fit(X_train, y_train)

# =========================================================
# 5️⃣ EVALUATION
# =========================================================

y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {acc*100:.2f}%\n")

# Classification report
report = classification_report(y_test, y_pred)
print("📊 Classification Report:\n")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=[0,1], yticklabels=[0,1]
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(pipe, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# =========================================================
# 6️⃣ EXAMPLE PREDICTIONS
# =========================================================

print("\n🎯 Example Predictions:")

example_data = pd.DataFrame([
    {
        'batting_team': 'Mumbai Indians',
        'bowling_team': 'Chennai Super Kings',
        'city': 'Mumbai',
        'runs_left': 50,
        'balls_left': 30,
        'wickets_left': 5,
        'target': 180,
        'crr': 8.0,
        'rrr': 10.0,
        'home_advantage': 1
    },
    {
        'batting_team': 'Royal Challengers Bangalore',
        'bowling_team': 'Kolkata Knight Riders',
        'city': 'Bangalore',
        'runs_left': 30,
        'balls_left': 12,
        'wickets_left': 3,
        'target': 150,
        'crr': 7.5,
        'rrr': 15.0,
        'home_advantage': 1
    },
    {
        'batting_team': 'Delhi Capitals',
        'bowling_team': 'Sunrisers Hyderabad',
        'city': 'Delhi',
        'runs_left': 70,
        'balls_left': 36,
        'wickets_left': 6,
        'target': 200,
        'crr': 6.5,
        'rrr': 11.7,
        'home_advantage': 1
    },
    {
        'batting_team': 'Kings XI Punjab',
        'bowling_team': 'Rajasthan Royals',
        'city': 'Pune',
        'runs_left': 20,
        'balls_left': 18,
        'wickets_left': 2,
        'target': 160,
        'crr': 8.0,
        'rrr': 6.7,
        'home_advantage': 0
    },
    {
        'batting_team': 'Chennai Super Kings',
        'bowling_team': 'Mumbai Indians',
        'city': 'Chennai',
        'runs_left': 100,
        'balls_left': 60,
        'wickets_left': 7,
        'target': 210,
        'crr': 7.0,
        'rrr': 10.0,
        'home_advantage': 1
    }
])

example_data['win_prob'] = pipe.predict_proba(example_data)[:,1]
example_data['predicted_result'] = pipe.predict(example_data)

for idx, row in example_data.iterrows():
    print(f"\nScenario {idx+1}:")
    print(f"Batting Team: {row['batting_team']}")
    print(f"Bowling Team: {row['bowling_team']}")
    print(f"City: {row['city']}")
    print(f"Runs Left: {row['runs_left']}")
    print(f"Balls Left: {row['balls_left']}")
    print(f"Wickets Left: {row['wickets_left']}")
    print(f"Win Probability: {row['win_prob']*100:.2f}%")
    print(f"Prediction: {'Win' if row['predicted_result']==1 else 'Lose'}")

# =========================================================
# # 7️⃣ SAVE MODEL
# # =========================================================
# joblib.dump(pipe, 'pipe.joblib')
# print("\n✅ Model saved as 'pipe.joblib'")
