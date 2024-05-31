from linear_reg_model import Linear_Reg_Model
import joblib
import json

def main(event, context):
    # message = "Hello World from the Kyma Function "+context['function-name']+" running on "+context['runtime']+ "!";
    # requestData = event['extensions']['request']


    with open('./sample_data.json', 'r') as file:
        json_data = json.load(file)

    loaded_model = joblib.load("linear_model_model.pkl")
    model = Linear_Reg_Model(loaded_model)
    results = model.predict(json_data)
    print(results)
    result = {'predictions': results.tolist()}

    return results
