import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, Y = None, None

def feature_importances_from_tree_based_model_():
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2, stratify=Y, random_state=123
    )

    model = RandomForestClassifier()
    model = model.fit(X_train, Y_train)

    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()

    return model