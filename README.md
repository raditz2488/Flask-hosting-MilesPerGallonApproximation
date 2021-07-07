# Flask-hosting-MilesPerGallonApproximation
This project demonstrates hosting a Flask app used to approximate the value of miles per gallon, given a few features of a car.
This app uses a trained neural network to approximate the miles per gallon(mpg).

# Files used:
1. NeuralNetworkModelCode.ipynb - For creation of neural network model to approximate mpg
2. CarsMPG.py - The file to start Flask app providing an enpoint to predict mpg
3. LoadModelTester.ipynb - Used totest loading of model from file.
4. EndpointTester.ipynb - Used to test the endpoint of the flask app
5. auto-mpg.csv - The dataset used to train the neural network for mpg approximation.

# How to run the flask app?
- Move to the root directory of the repository.
- Launch the terminal and run the flask app by using the command in terminal: python CarsMPG.py
- After the run the terminal will display the url at which the app is hosted.
- This url can be used to hit the '/api' route.
