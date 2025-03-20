def main(df):

    import h2o
    import pandas as pd
    from h2o.automl import H2OAutoML
    from h2o.frame import H2OFrame
    from h2o.estimators import H2OStackedEnsembleEstimator

    # Initialize H2O
    h2o.init()

    # Loading input pandas DataFrame
    pandas_df = df

    # Converting pandas DataFrame to H2O Frame
    data = h2o.H2OFrame(pandas_df)

    # Setting the threshold value for classification
    threshold_value = 1.1
    response = "USPTailing"

    # Binary classification based on the threshold value
    data['result'] = (data[response] <= threshold_value).ifelse("Pass", "Fail")
    
    
    # Function to split dataset
    def split_dataset(data, train_ratio=0.7, val_ratio=0.1):
        train, temp = data.split_frame(ratios=[train_ratio])
        valid, test = temp.split_frame(ratios=[val_ratio / (1 - train_ratio)])
        return train, valid, test

    train_ratio = 0.7
    valid_ratio = 0.1
    train, valid, test = split_dataset(data, train_ratio, valid_ratio)

    predictors = data.columns
    predictors.remove(response)
    predictors.remove('result')

    aml = H2OAutoML(max_models=20, seed=1, nfolds=5, keep_cross_validation_predictions=True)
    aml.train(x=predictors, y='result', training_frame=train, validation_frame=valid)

    lb = aml.leaderboard
    lb_df = lb.as_data_frame()

    # Add additional metrics for train, validation, and test data
    def get_metrics(model, data, label):
        performance = model.model_performance(test_data=data)
        metrics = {
            f'{label}_auc': performance.auc(),
            f'{label}_precision': performance.precision()[0][1],
            f'{label}_f1': performance.F1()[0][1]
        }
        return metrics

    for model_id in lb['model_id'].as_data_frame()['model_id']:
        model = h2o.get_model(model_id)
        train_metrics = get_metrics(model, train, 'train')
        valid_metrics = get_metrics(model, valid, 'valid')
        test_metrics = get_metrics(model, test, 'test')
        for key, value in {**train_metrics, **valid_metrics, **test_metrics}.items():
            lb[lb['model_id'] == model_id, key] = value

    print(lb)

    #Getting the best models from the model IDs (in this case the top 3)
    best_models = [h2o.get_model(model_id) for model_id in lb_df['model_id'][:20]]

    for model in best_models:
        print(f"Model ID: {model.model_id}")
        print(f"Cross-validation predictions: {model._model_json['output']['cross_validation_predictions'] is not None}")
        train_performance = model.model_performance(train)
        valid_performance = model.model_performance(valid)
        test_performance = model.model_performance(test)
        
        print(f"Train metrics for {model.model_id}:")
        print(f"AUC: {train_performance.auc()}")
        print(f"Precision: {train_performance.precision()[0][1]}")
        print(f"F1: {train_performance.F1()[0][1]}")
        
        print(f"Validation metrics for {model.model_id}:")
        print(f"AUC: {valid_performance.auc()}")
        print(f"Precision: {valid_performance.precision()[0][1]}")
        print(f"F1: {valid_performance.F1()[0][1]}")
        
        print(f"Test metrics for {model.model_id}:")
        print(f"AUC: {test_performance.auc()}")
        print(f"Precision: {test_performance.precision()[0][1]}")
        print(f"F1: {test_performance.F1()[0][1]}")

    # Filtering out models that do not have cross-validation predictions
    best_models = [model for model in best_models if model._model_json['output']['cross_validation_predictions'] is not None]

    # Printing model id of each filtered model
    for model in best_models:
        print(f"Filtered Model ID: {model.model_id}")
        
    #Creating an ensemble of the best models

    
    if best_models:
        ensemble = H2OStackedEnsembleEstimator(
            base_models=[model.model_id for model in best_models],
            metalearner_algorithm="AUTO"
        )
        ensemble.train(x=predictors, y='result', training_frame=train)

        # Evaluate the ensemble model
        predictions = ensemble.predict(test[predictors])
        test_with_preds = test.cbind(predictions)

        print("Actual vs Predicted:")
        print(test_with_preds[['result', "predict"]].head())

    # Convert H2O Frame to pandas DataFrame
    pandas_test_with_preds = test_with_preds.as_data_frame()

    # Convert pandas DataFrame to JSON string
    json_result = pandas_test_with_preds.to_json(orient='records')
    
    return json_result