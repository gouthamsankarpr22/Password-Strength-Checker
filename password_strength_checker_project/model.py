import numpy as np,pandas as pd, random, string, math
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset_from_csv(filepath="passwords.csv"):
    """
    Loads passwords and labels from a CSV file.
    Assumes CSV has 'password' and 'strength' columns.
    """
    try:
        # 1. Read 'strength' as a string (object) first to avoid errors
        df = pd.read_csv(filepath,
                         usecols=['password','strength'],
                         dtype={'password': str, 'strength': str}, 
                         on_bad_lines='skip')
        
        # 2. Try to convert 'strength' to a number. 
        #    errors='coerce' will turn any bad values (like "Weak") into NaN (Not a Number)
        df['strength'] = pd.to_numeric(df['strength'], errors='coerce')

        # 3. Drop any rows that had bad data (NaN) or missing passwords
        df = df.dropna(subset=['password', 'strength'])
        
        pwds = df['password'].tolist()
        # 4. Now that it's clean, convert the strength column to integer
        labels = df['strength'].astype(int).tolist()
        
        print(f"Loaded {len(pwds)} valid passwords from {filepath}")
        return pwds, labels
        
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        raise
    except KeyError:
        print("Error: CSV file must include 'password' and 'strength' columns.")
        raise
        
def extract_features(password):
    specials = set('!@#$%^&*()_+-=[]{};:,.<>/?|')
    length = len(password)
    digits = sum(c.isdigit() for c in password)
    upper = sum(c.isupper() for c in password)
    lower = sum(c.islower() for c in password)
    symbols = sum(c in specials for c in password)
    repeat = sum(1 for i in range(1,len(password)) if password[i]==password[i-1])
    entropy_est = 0.0
    if length>0:
        charset = 0
        if any(c.islower() for c in password): charset += 26
        if any(c.isupper() for c in password): charset += 26
        if any(c.isdigit() for c in password): charset += 10
        if any(c in specials for c in password): charset += len(specials)
        if charset>0:
            entropy_est = length * math.log2(charset)
    has_mixed = 1 if upper>0 and lower>0 else 0
    has_digit = 1 if digits>0 else 0
    has_symbol = 1 if symbols>0 else 0
    return [length, digits, upper, lower, symbols, repeat, int(entropy_est), has_mixed, has_digit, has_symbol]

# REPLACE your old train_model function with this one
def train_model():
    # Load data from CSV instead of generating it
    pwds, labels = load_dataset_from_csv("passwords.csv") 

    if not pwds:
        # This will be caught by the exception in app.py
        raise ValueError("No data loaded from CSV. Training cannot proceed.")

    X = np.array([extract_features(pw) for pw in pwds])
    y = np.array(labels)
    
    scaler = StandardScaler()
    # Check if X is empty or has only one sample
    if X.shape[0] < 2:
        raise ValueError("Not enough data to train the model after feature extraction.")

    Xs = scaler.fit_transform(X)
    
    # Check if we have enough samples for splitting
    # stratify needs at least 2 members in each class for a 0.2 test split
    if X.shape[0] < 5: 
        # Not enough data to split, train on all of it
        X_train, y_train = Xs, y
        # Create a dummy test set to calculate 'accuracy'
        X_test, y_test = Xs, y 
    else:
        # Ensure all classes are present for stratification
        unique_labels, counts = np.unique(y, return_counts=True)
        # Find labels with only one sample
        single_sample_labels = unique_labels[counts == 1]
        
        if len(single_sample_labels) > 0 or len(unique_labels) < 3:
            # If stratification is not possible, just do a regular split
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42)
        else:
            # Proceed with stratified split
            X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    return clf, scaler, acc