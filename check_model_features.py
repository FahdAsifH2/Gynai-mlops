import joblib

# Load model
model = joblib.load("models/model.pkl")

# Get feature names from the final estimator (it's a StackingClassifier)
if hasattr(model, 'feature_names_in_'):
    features = model.feature_names_in_
    print("Found feature names!")
    print("\nCopy this list into your Flask API:\n")
    print("feature_order = [")
    for feature in features:
        print(f'    "{feature}",')
    print("]")
else:
    print("Couldn't find feature names in model")