import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error



def load_models():
    try:
        lgb_model = joblib.load('src/models/lgb_model.joblib')
        print("LightGBM model loaded successfully.")
    except FileNotFoundError:
        print("LightGBM model not found.")
        lgb_model = None

    return lgb_model



def evaluate_model(model, X_test, y_test, model_name="Model"):
    if model is None:
        print(f"{model_name} is not available for evaluation.")
        return

    predictions = model.predict(X_test)

    mAE = mean_absolute_error(y_test, predictions)
    mSE = mean_squared_error(y_test, predictions)
    rmSE = mean_squared_error(y_test, predictions, squared=False)

    print(f"\nEvaluation for {model_name}:")
    print(f"  Mean Absolute Error (MAE): {mAE:.4f}")
    print(f"  Mean Squared Error (MSE): {mSE:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmSE:.4f}")
    print("-" * 50)



def plot_learning_curve(model, X_train, y_train, X_val, y_val, model_name="Model"):
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []

    for m in train_sizes:
        model.fit(X_train[:int(m * len(X_train))], y_train[:int(m * len(y_train))])
        train_scores.append(mean_squared_error(y_train[:int(m * len(y_train))], model.predict(X_train[:int(m * len(X_train))]), squared=False))
        val_scores.append(mean_squared_error(y_val, model.predict(X_val), squared=False))

    plt.plot(train_sizes, train_scores, label="Training Error")
    plt.plot(train_sizes, val_scores, label="Validation Error")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE")
    plt.title(f"Learning Curve - {model_name}")
    plt.legend()
    plt.show()



X_train_full = np.random.rand(100, 4)
y_train_full = np.random.rand(100) * 1000

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)




lgb_model= load_models()


evaluate_model(lgb_model, X_val, y_val, model_name="LightGBM")

# Fine-tuning and learning curve for LightGBM
lgb_model.set_params(learning_rate=0.05, max_depth=6, n_estimators=100)



plot_learning_curve(lgb_model, X_train, y_train, X_val, y_val, "LightGBM")

