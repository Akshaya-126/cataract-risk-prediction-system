from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("cataract_risk_model.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
le_target = data['le_target']

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    reason = None
    recommendation = None

    if request.method == "POST":
        # -------------------------
        # Get form inputs
        # -------------------------
        age = int(request.form['age'])
        blurred_vision = int(request.form['blurred_vision'])
        glare = int(request.form['glare'])
        night_vision_issue = int(request.form['night_vision_issue'])

        smoking_status = request.form['smoking_status']
        family_history = request.form['family_history']
        diabetes = request.form['diabetes']

        # -------------------------
        # Encode categorical values
        # -------------------------
        mapping_smoke = {'Never': 0, 'Former': 1, 'Current': 2}
        mapping_family = {'No': 0, 'Yes': 1}
        mapping_diabetes = {'Non-diabetic': 0, 'Diabetic': 1}

        smoke_val = mapping_smoke[smoking_status]
        family_val = mapping_family[family_history]
        diabetes_val = mapping_diabetes[diabetes]

        # -------------------------
        # 🔑 IMPORTANT: CREATE FEATURES USED IN TRAINING
        # -------------------------

        # age_bin (same logic as training)
        if age < 40:
            age_bin = 0
        elif age < 60:
            age_bin = 1
        else:
            age_bin = 2

        # symptom_score (same logic as training)
        symptom_score = (
            blurred_vision +
            glare +
            night_vision_issue +
            diabetes_val
        )

        # -------------------------
        # Create input DataFrame (MATCH FEATURE NAMES)
        # -------------------------
        x_input = pd.DataFrame([[
            age_bin,
            symptom_score,
            smoke_val,
            family_val,
            diabetes_val
        ]], columns=[
            'age_bin',
            'symptom_score',
            'Smoking Status',
            'Family History',
            'Diabetes'
        ])

        # -------------------------
        # Predict
        # -------------------------
        pred_enc = model.predict(x_input)[0]
        result = le_target.inverse_transform([pred_enc])[0]

        # -------------------------
        # Reason text
        # -------------------------
        reasons = []
        if blurred_vision: reasons.append("Blurred vision")
        if glare: reasons.append("Glare")
        if night_vision_issue: reasons.append("Night vision issues")
        if diabetes == "Diabetic": reasons.append("Diabetes")
        if age >= 60: reasons.append("Age above 60")

        reason = ", ".join(reasons) if reasons else "No major symptoms"

        # -------------------------
        # Recommendation
        # -------------------------
        if result == "High":
            recommendation = "⚠️ High risk detected. Please consult an eye specialist immediately."
        elif result == "Medium":
            recommendation = "⚠️ Moderate risk detected. Consider a clinical eye checkup."
        else:
            recommendation = "✅ Low risk. Maintain regular eye checkups."

    return render_template(
        "index.html",
        result=result,
        reason=reason,
        recommendation=recommendation
    )

if __name__ == "__main__":
    app.run(debug=True)