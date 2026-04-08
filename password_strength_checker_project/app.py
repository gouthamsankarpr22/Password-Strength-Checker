from flask import Flask, render_template, request, jsonify
from model import train_model, extract_features
import numpy as np, threading
import csv

app = Flask(__name__)
model = None
scaler = None
train_info = {"done": False, "accuracy": None, "message": "Training not started"}

# REPLACE your old function with this one
def load_common_passwords(filepath="common_passwords.csv"):
    """Loads a list of common passwords from a one-column CSV file."""
    common_set = set()
    try:
        # --- THIS IS THE FIX ---
        # Change 'utf-8' to 'utf-8-sig'
        # 'utf-8-sig' automatically handles and removes the invisible BOM
        with open(filepath, 'r', encoding='utf-8-sig') as f:
        # --- END OF FIX ---
            reader = csv.reader(f)
            for row in reader:
                if row: # Check if row is not empty
                    # Add the password (from the first column) in lowercase
                    common_set.add(row[0].strip().lower())
        print(f"Loaded {len(common_set)} common passwords from {filepath}")
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. No common password blocklist loaded.")
    except Exception as e:
        print(f"Error loading {filepath}: {e}. Using empty list.")
    
    return common_set

# Load the list on startup

def background_train():
    global model, scaler, train_info
    train_info["message"] = "Training model... please wait"
    try:
        m, s, acc = train_model()
        model, scaler = m, s
        train_info["done"] = True
        train_info["accuracy"] = acc
        train_info["message"] = f"Training complete (accuracy={acc:.3f})"
    except Exception as e:
        train_info["message"] = f"Error: {e}"

threading.Thread(target=background_train, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html", train_info=train_info)

@app.route("/predict", methods=["POST"])
def predict():
    if not train_info["done"]:
        return jsonify({"error":"Model still training"}), 503
    pw = request.form.get("password","")
    if pw.strip() == "":
        return jsonify({"error":"Empty password"}), 400
    feats = extract_features(pw)
    X = np.array(feats).reshape(1,-1)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    probs = model.predict_proba(Xs)[0].tolist()
    label_map = {0:"Weak",1:"Medium",2:"Strong"}
    return jsonify({"prediction": label_map[pred], "probabilities": {"Weak":probs[0],"Medium":probs[1],"Strong":probs[2]}})
if __name__ == "__main__":
    app.run(debug=True)
