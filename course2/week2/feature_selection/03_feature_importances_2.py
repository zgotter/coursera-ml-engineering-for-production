from sklearn.feature_selection import SelectFromModel

df = None

def select_features_from_model(model):
    model = SelectFromModel(model, prefit=True, threshold=0.012)

    feature_idx = model.get_support()
    feature_names = df.drop('diagnosis_int', 1).columns[feature_idx]
    return feature_names