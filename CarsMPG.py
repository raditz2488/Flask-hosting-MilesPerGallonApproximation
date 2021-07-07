from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
import uuid

app = Flask(__name__)

model = load_model('mpg_model.h5')

Expected = {
"cylinders":{"min":3,"max":8},
"displacement":{"min":68.0,"max":455.0},
"horsepower":{"min":46.0,"max":230.0},
"weight":{"min":1613,"max":5140},
"acceleration":{"min":8.0,"max":24.8},
"year":{"min":70,"max":82},
"origin":{"min":1,"max":3}
}

@app.route('/api', methods=['POST'])
def mpg_prediction():
    errors = []

    content = request.json

    for name in content:
        if name in Expected:
            # If name is in Expected we compare if the value is in min and max range
            max_val = Expected[name]['max']
            min_val = Expected[name]['min']
            
            val = content[name]

            if val < min_val or val > max_val:
                errors.append(f'Value for {name}, should be between {min_val} and {max_val}')
        else:
            errors.append(f'Unexpected field {name}')

    for name in Expected:
        if name not in content:
            errors.append(f'Missing value: {name}')

    if len(errors) < 1:
        x= np.zeros((1,7))
        
        x[0,0] = content['cylinders']
        x[0,1] = content['displacement']
        x[0,2] = content['horsepower']
        x[0,3] = content['weight']
        x[0,4] = content['acceleration']
        x[0,5] = content['year']
        x[0,6] = content['origin']

        prediction = model.predict(x)
        mpg = float(prediction[0])
        response = {'id': uuid.uuid4(), 'mpg': mpg, 'errors': errors}
    else:
        response = {'id': uuid.uuid4(), errors: errors}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
        