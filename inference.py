import joblib, numpy as np, os, json

def model_fn(model_dir):
    return {
        'model':    joblib.load(os.path.join(model_dir, 'model.joblib')),
        'imputer':  joblib.load(os.path.join(model_dir, 'imputer.joblib')),
        'features': joblib.load(os.path.join(model_dir, 'feature_cols.joblib'))
    }

def input_fn(request_body, content_type):
    if content_type == 'application/json':
        data = json.loads(request_body)
        return np.array(data)
    raise ValueError(f'Unsupported content type: {content_type}')

def predict_fn(input_data, artifacts):
    X     = input_data.reshape(1, -1)
    X_imp = artifacts['imputer'].transform(X)
    pred  = artifacts['model'].predict(X_imp)[0]
    proba = artifacts['model'].predict_proba(X_imp)[0]
    labels = ['Normal', 'Osteopenia', 'Osteoporosis']
    return {
        'prediction':    labels[int(pred)],
        'confidence':    float(proba[int(pred)]),
        'probabilities': dict(zip(labels, proba.tolist()))
    }

def output_fn(prediction, accept):
    return json.dumps(prediction), 'application/json'