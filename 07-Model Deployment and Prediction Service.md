Deploying the model : making the model "running and accessible"

easy part :
```
# Example of how to use FastAPI to turn your predict function
# into a POST endpoint
@app.route('/predict', methods=['POST'])
def predict():
  X = request.get_json()['X']
  y = MODEL.predict(X).tolist()
  return json.dumps({'y': y}), 200
```

hard part :
- making the model available to millions of users with a latency of milliseconds and 99% uptime.
- setting up the infrastructure so that the right person can be immediately notified when something goes wrong
- deploying the updates to fix what’s wrong

Deployements in some companies :
- ppl who developed the model
- a separated team



Machine Learning Deployment Myths : 
- You Only Deploy One or Two ML Models at a Time : Uber (ride demand, driver availability, estimated time of
arrival, dynamic pricing, fraudulent transaction, customer churn,...)
- If We Don’t Do Anything, Model Performance Remains the Same : data distribution shifts problems
- You Won’t Need to Update Your Models as Much (udpdating the model is du to its performance decay)
- Most ML Engineers Don’t Need to Worry About Scale (number of users -> many users means many
- requests for the model -> it might take time + new distribution) this is a problem mostly for high tech corproations + [edge]

Batch Prediction Versus Online Prediction
3 modes of prediction
- Batch prediction, which uses only batch features.
- Online prediction that uses only batch features (e.g., precomputed embeddings).
- Online prediction that uses both batch features and streaming features. This is also known as streaming prediction.

Online prediction is when predictions are generated and returned as soon as requests for these predictions arrive. (Google translate)
Traditionally : when doing online prediction, requests are sent to the prediction service via RESTful APIs (predictions are generated in synchronization with requests -> synchronous prediction)

Batch prediction is when predictions are generated periodically or whenever triggered. The predictions are stored somewhere, such as in SQL tables or an in-memory database, and retrieved as needed.
Netflix might generate movie recommendations for all of its users every four hours.. (asynchronous prediction)

**note : ** Both online prediction and batch prediction can make predictions for multiple samples (in batch) or one at a time only one of them is synchrounous and the other is asynchrounous

Batch prediction : 
- Frequency : Periodical, such as every four hours
- Useful for : Processing accumulated data when you don’t need immediate results (such as recommender systems)
- Optimized for : High throughput

Online prediction:
- As soon as requests come
- When predictions are needed as soon as a data sample is generated (such as fraud detection)
- Low latency

Restuarant application can use both online & batch predictions 
batch : recommendation system each 3h
online : when clicking a restaurant, a new recommendation for similar restaurant is generated as soon as the request is made

Many ppl believe that online prediction is less efficient : both in terms of cost and performance
this is not necessarly true : 2% of users could be using the app

From Batch Prediction to Online Prediction
- more natural way to serve prediction is prb online
- problem, might take too long : what if we pregenerated the translation before it gets computed
- predictions are precomputed : doesn't matter how long they take
- using batch prediction to make a set of predictions to reduce inference latency
- problem, less responsive to user's change of preferences -> horror movie to comedy movie recommendation in netflix
- Another problem, what are requests to generate predictions in advance
- Batch prediction could lead to catastrophic failures : high freq trading, autonomus veihcules, voice assistants, unlocking the phone using fingerprints
- Hardware is being developed to allow faster cheaper online prediction : "why precompute & store in db predictions that are not particularly correct or could lead to desasters in term of business"
- How to overcome latency challenge in online predictions
  - A (near) realtime pipeline that can work with incoming data, extract streaming feature, putting them into a model & return the prediction in near realtime
  - A model that can generate prediction at a speed acceptable to its end users


Unifying Batch Pipeline and Streaming Pipeline
