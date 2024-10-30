# runtime_v_training
This repository provides a simple contrast between a a basic model's training then prediction for analysis versus the same trained model's usage in a runtime.

# Premise
Documentation, particularly around AI platforms, often implies that a model can be trained and immediately available for prediction.

In a technical sense, this is true; but establishing an AI solution requires additional software engineering to strengthen the user experience (e.g. by handling errors elegantly), increase resiliency (e.g. introducing monitoring) and support scale. 

This repository provides a very simple contrast between training an extremely simple classification model and allowing it to operate in a runtime.  It is quickly evident that the process for training and prediction does not support the quick, simple interaction a user would expect.

> NOTE: This training and model in this repository are purely for illustrative purposes. The dataset contains apparent bias; and the model's performance deserves additional attention to the quality of its predictions. This is intentional as the goal is to illustrate model training vs prediction without delving into the data sicnece itself. 

# Using this repository
You can use this repository in a few ways:
1. Run the solution as-is. You can run the training script then run the fastAPI application with minimal environmental set up. You will what it means to first train then serve predictions from the resultant model.
1. Modify the model... You will notice the model training is extremely basic. Its base accuracy is around 70%; but you can adjust the training to achieve improved results. Completed training will export the model for use to serve predictions.
1. Use the software components... The docker file and app.py files provide a rough skeleton for serving predictions in a cloud/docker-based environment. As a result, you can use the code in this repository as examples o starting points for your own code.

# Run the code locally
## Prerequisites
1. You will need git installed on your local machine. The simplest means to accomplish this is installing GitHub [Desktop](https://desktop.github.com/download/).
2. You will want a IDE to work in with the code. The source code was originally written and tested in [VS Code](https://code.visualstudio.com/), although any IDE suporting a python development environment will allow youto interact with the code.
3. Download the related dataset [here](https://www.kaggle.com/datasets/darelljohnson/sample-logistical-reg/data) and add it to the data folder of this repository once you have cloned it..
## Establish your python environment
1. Clone the repository to your local machine.
2. Create a virtual environment by running the following command: ```python -m venv env```
3. Activate your enviroment with this command: ```source ./env/bin/activate```
4. Make sure your pip installation is up to date: ```python -m pip install --upgrade pip```
5. Install the application dependencies: ```pip install -r ./src/requirements.txt```

## Run the code
This code can operate in 3 primary ways:

1. To run training, execute the command: ```python3 ./src/training.py ```. 
> This is an optional step; but it is an interesting one allowing you to observe the timing. Some model development patterns will fit/train a model prior to serving predictions; but this leads to a significant delay between request a prediction and seeing it served.

2. To run the python application and serve predictions, execute the unit tests using this command: ```pytest src/test.py --log-cli-level=debug```

3. It is also possible to build and run the application within its docker container as a fastAPI application. To do this, use docker build to build then container then run it. 

> NOTE: The primary aim for this repository is educating those seeking to begin understanding a basic process to translate an AI model into a runtime. As a result, detailed instructions for using docker are being omitted.
