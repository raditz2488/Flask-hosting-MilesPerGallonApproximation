from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify, render_template
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

@app.route('/', methods=['GET'])
def index():
    return render_template('ImageML.html')

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
            # The feature name is unexpected 
            errors.append(f'Unexpected field {name}')

    for name in Expected:
        if name not in content:
            # The feature name is missing from the content in request
            errors.append(f'Missing value: {name}')

    if len(errors) < 1:
        # If we have zero errors that means we are in a good state to predict
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
        val = f'{mpg} mpg'
    else:
        response = {'id': uuid.uuid4(), errors: errors}
        val = errors[0]

    # Return the appropriate response generated in one of the flows above.
    return render_template('ImageML.html', prediction=val)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
        