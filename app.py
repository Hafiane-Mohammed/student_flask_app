from flask import Flask, render_template, request
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Charger le modèle et les colonnes utilisées
model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))

# Charger dataset pour dashboard/statistiques
df = pd.read_csv("students.csv")
df.columns = df.columns.str.replace(" ", "_")

# Créer le dossier plots si inexistant
if not os.path.exists("static/plots"):
    os.makedirs("static/plots")

# Fonction pour générer graphiques
def generate_plots():
    # Score distribution
    plt.figure(figsize=(6,4))
    plt.hist([df['math_score'], df['reading_score'], df['writing_score']],
             bins=15, label=['Math','Reading','Writing'], alpha=0.7)
    plt.legend()
    plt.title("Score Distribution")
    plt.tight_layout()
    plt.savefig("static/plots/score_distribution.png")
    plt.close()

    # Gender distribution
    plt.figure(figsize=(6,4))
    df['gender'].value_counts().plot(kind='bar', color=['#764ba2','#667eea'], title="Gender Distribution", rot=0)
    plt.tight_layout()
    plt.savefig("static/plots/gender_distribution.png")
    plt.close()

    # Lunch type distribution
    plt.figure(figsize=(6,4))
    df['lunch'].value_counts().plot(kind='bar', color=['#f8cdda','#1d2b64'], title="Lunch Distribution", rot=0)
    plt.tight_layout()
    plt.savefig("static/plots/lunch_distribution.png")
    plt.close()

    # Reading vs Math correlation
    plt.figure(figsize=(6,4))
    plt.scatter(df['reading_score'], df['math_score'], c='orange')
    plt.xlabel('Reading Score')
    plt.ylabel('Math Score')
    plt.title('Reading vs Math Correlation')
    plt.tight_layout()
    plt.savefig("static/plots/reading_math_corr.png")
    plt.close()

# Générer les plots au lancement
generate_plots()

# Route page accueil
@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

# Route prédiction
@app.route("/predict", methods=["POST"])
def predict():
    gender_input = request.form['gender']
    lunch_input = request.form['lunch']
    test_prep_input = request.form['test_preparation_course']
    reading = max(0.0, min(100.0, float(request.form['reading_score'])))
    writing = max(0.0, min(100.0, float(request.form['writing_score'])))

    # Encodage 0/1 comme dans le modèle
    gender = 1 if gender_input == "male" else 0
    lunch = 1 if lunch_input == "standard" else 0
    test_prep = 1 if test_prep_input == "completed" else 0

    df_input = pd.DataFrame([[gender, lunch, test_prep, reading, writing]], columns=model_columns)
    prediction = model.predict(df_input)[0]
    
    # Restreindre la prédiction entre 0 et 100
    prediction = max(0.0, min(100.0, prediction))

    # Conseils dynamiques
    advice = []
    if test_prep_input == 'none':
        advice.append("Nous recommandons de suivre le cours de préparation.")
    if reading < 60:
        advice.append("Améliorez la lecture pour booster les notes de math.")

    return render_template("result.html", prediction=round(prediction,2), advice=advice)

# Route dashboard
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Route statistiques globales
@app.route("/stats")
def stats():
    # Moyenne, mediane, min, max par gender
    gender_stats = df.groupby('gender')['math_score'].agg(['mean','median','min','max']).reset_index()
    # Moyenne, mediane, min, max par lunch
    lunch_stats = df.groupby('lunch')['math_score'].agg(['mean','median','min','max']).reset_index()
    return render_template("stats.html", gender_stats=gender_stats, lunch_stats=lunch_stats)

if __name__ == "__main__":
    app.run(debug=True)