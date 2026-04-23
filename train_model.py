import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Charger les données
df = pd.read_csv("students.csv")
df.columns = df.columns.str.replace(" ", "_")

# Créer dossier plots si inexistant
if not os.path.exists("static/plots"):
    os.makedirs("static/plots")

# Encodage pour ML
X = df[['gender','lunch','test_preparation_course','reading_score','writing_score']]
X['gender'] = X['gender'].replace({'male':1,'female':0})
X['lunch'] = X['lunch'].replace({'standard':1,'free/reduced':0})
X['test_preparation_course'] = X['test_preparation_course'].replace({'completed':1,'none':0})
y = df['math_score']

# Split pour modèle (pas de StandardScaler, inutile pour RandomForest et compatible avec app.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Liste des modèles à tester
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_r2 = -float("inf")
best_name = ""

print("\n--- Comparaison des modèles ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{name:<20} -> MSE: {mse:.2f}, R2: {r2:.4f}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_name = name

print(f"\n=> Le meilleur modèle est '{best_name}' avec un R2 de {best_r2:.4f}")

# --- Sauvegarde du meilleur modèle pour l'application Flask ---
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(list(X.columns), open("columns.pkl", "wb"))
print(f"Modèle '{best_name}' sauvegardé dans model.pkl et columns.pkl.\n")

# --- Fonction de prédiction pour test interne ---
def predict_score(gender, lunch, test_prep, reading, writing):
    df_in = pd.DataFrame([[gender, lunch, test_prep, reading, writing]], columns=X.columns)
    return best_model.predict(df_in)[0]

print("Test de prédiction interne ({}) : {:.2f}".format(best_name, predict_score(1, 1, 1, 80, 85)))

# --- 1. Score Distribution ---
plt.figure(figsize=(6,4))
plt.hist([df['math_score'], df['reading_score'], df['writing_score']], bins=15, label=['Math','Reading','Writing'], alpha=0.7)
plt.legend()
plt.title("Score Distribution")
plt.xlabel("Scores")
plt.ylabel("Number of Students")
plt.text(5, 15, "Histogramme montrant la répartition des scores", fontsize=9, color='black')
plt.savefig("static/plots/score_distribution.png")
plt.close()

# --- 2. Gender Distribution ---
plt.figure(figsize=(6,4))
counts = df['gender'].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette=['#764ba2','#667eea'])
plt.title("Gender Distribution")
plt.ylabel("Number of Students")
plt.text(0, max(counts.values)/2, "Répartition garçons vs filles", fontsize=9)
plt.savefig("static/plots/gender_distribution.png")
plt.close()

# --- 3. Lunch Distribution ---
plt.figure(figsize=(6,4))
counts = df['lunch'].value_counts()
sns.barplot(x=counts.index, y=counts.values, palette=['#f8cdda','#1d2b64'])
plt.title("Lunch Type Distribution")
plt.ylabel("Number of Students")
plt.text(0, max(counts.values)/2, "Type de repas des étudiants", fontsize=9, color='white')
plt.savefig("static/plots/lunch_distribution.png")
plt.close()

# --- 4. Reading vs Math Correlation ---
plt.figure(figsize=(6,4))
plt.scatter(df['reading_score'], df['math_score'], c='orange')
plt.xlabel('Reading Score')
plt.ylabel('Math Score')
plt.title('Reading vs Math Correlation')
plt.text(50, max(df['math_score'])-10, "Les étudiants qui lisent mieux ont souvent de meilleures notes en maths", fontsize=9)
plt.savefig("static/plots/reading_math_corr.png")
plt.close()

# --- Feature Importance / Coefficients ---
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = best_model.coef_
else:
    importances = [0] * len(X.columns)

plt.figure(figsize=(6,4))
plt.bar(X.columns, importances)
plt.title(f"Feature Importance ({best_name})")
plt.ylabel("Importance Score / Coefficient")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("static/plots/feature_importance.png")
plt.close()

# --- 6. Correlation Heatmap ---
plt.figure(figsize=(6,4))
sns.heatmap(df[['math_score','reading_score','writing_score']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("static/plots/corr_heatmap.png")
plt.close()

# --- 7. Boxplots by Gender ---
plt.figure(figsize=(6,4))
sns.boxplot(x='gender', y='math_score', data=df, palette=['#764ba2','#667eea'])
plt.title("Math Score by Gender")
plt.text(0, max(df['math_score'])-10, "Dispersion des notes par genre", fontsize=9)
plt.savefig("static/plots/box_gender.png")
plt.close()

# --- 8. Boxplots by Lunch ---
plt.figure(figsize=(6,4))
sns.boxplot(x='lunch', y='math_score', data=df, palette=['#f8cdda','#1d2b64'])
plt.title("Math Score by Lunch Type")
plt.savefig("static/plots/box_lunch.png")
plt.close()