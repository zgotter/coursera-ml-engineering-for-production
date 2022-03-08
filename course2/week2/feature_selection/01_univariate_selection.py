from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

X, Y = None, None
df = None

def univariate_selection():
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=0.2, stratify=Y, random_state=123
    )
    
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    min_max_scaler = MinMaxScaler()
    Scaled_X = min_max_scaler.fit_transform(X_train_scaled)

    selector = SelectKBest(chi2, k=20) # Use Chi-Squared test
    X_new = selector.fit_transform(Scaled_X, Y_train)

    feature_idx = selector.get_support()
    feature_names = df.drop("diagnosis_int", axis=1).columns[feature_idx]
    return feature_names
