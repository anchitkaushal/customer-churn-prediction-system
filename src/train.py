def train(models,X_train,y_train):
   
    train_models = {}
   
    for name,model in models.items():
       model.fit(X_train,y_train)
       train_models[name] = model
   
    return train_models