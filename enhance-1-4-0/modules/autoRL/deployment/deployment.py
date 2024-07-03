# deployment/deployment.py

def deploy_model(model, model_path):
    model.save(model_path)
