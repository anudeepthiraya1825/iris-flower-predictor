from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from google.colab import files

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# spliting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "iris_model.pkl")


files.download("iris_model.pkl")
