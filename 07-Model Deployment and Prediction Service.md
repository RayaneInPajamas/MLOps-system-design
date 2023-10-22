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
# imgs
Model Compression:
- making the the model smaller : make it do inference faseter -> inference optimization
Techniques:

- Low-Rank Factorization: is a technique aimed at reducing the complexity of high-dimensional tensors by replacing them with lower-dimensional counterparts. One specific method of low-rank factorization involves the use of compact convolutional filters. In this approach, over-parameterized convolution filters, which tend to have an excessive number of parameters, are substituted with more compact blocks. This substitution serves a dual purpose: it decreases the total parameter count while also boosting computational efficiency

- Knowledge Distillation is a technique that involves training a smaller model, known as the "student," to replicate the behavior of a larger model or a group of models, referred to as the "teacher." The primary purpose of this approach is to deploy the smaller, more lightweight student model. While it's common for the student model to be trained after the teacher model, they can also be trained concurrently. A notable example of knowledge distillation in practical use is DistilBERT, which reduces the size of a BERT model by 40% while retaining 97% of its language understanding capabilities and providing a 60% speed improvement

- Pruning is a technique originally developed for decision trees, primarily to eliminate non-essential and redundant branches of the tree during classification. As neural networks gained wider acceptance, it became evident that these networks were often over-parameterized, leading to efforts to reduce the computational burden caused by the surplus parameters. In the context of neural networks, pruning encompasses two primary meanings. Firstly, it involves the removal of entire nodes or connections within a neural network, resulting in architectural changes and a reduction in the total number of parameters. The more common interpretation of pruning is the identification of parameters that contribute minimally to predictions and setting them to zero. This type of pruning does not alter the overall parameter count but decreases the number of non-zero parameters. It effectively makes the neural network more sparse, which typically demands less storage space compared to dense structures. Empirical experiments have demonstrated that pruning techniques can reduce the count of non-zero parameters in trained networks by over 90%, leading to reduced storage requirements and improved computational efficiency during inference, all without compromising the overall accuracy.

- Quantization is a widely employed and versatile model compression technique that finds extensive use in various tasks and neural network architectures. It focuses on reducing the size of a model by employing fewer bits to represent its parameters.As an example, floating-point numbers are represented with 32 bits for precision. For a model with 100 million parameters, each necessitating 32 bits, the storage requirement is 400 megabytes. By adopting 16-bit representation, referred to as half precision, we can reduce the memory footprint by half


## ML on the Cloud and on the Edge
Cloud Computing:

Involves performing a substantial portion of computations on remote cloud servers, either public or private.
Easy to deploy through managed cloud services like AWS or GCP, simplifying model production.
Downsides include high costs, especially for compute-intensive ML models. In 2018, some large companies were already spending hundreds of millions of dollars on cloud expenses yearly, while smaller firms could spend between $50K and $2M per year.
Mistakes in cloud service management can lead to financial difficulties or even bankruptcy for startups.
As cloud costs rise, companies are seeking ways to shift computations to edge devices to reduce server expenses.
Edge Computing:

Involves offloading a significant share of computations to consumer devices, including smartphones, laptops, cars, security cameras, robots, embedded devices, FPGAs, and ASICs.
Edge computing becomes appealing due to its cost-effective attributes and other features.
It enables applications to function where cloud computing is not feasible, including areas with unreliable or absent internet connections, rural regions, and places with strict no-internet policies.
Reduces concerns about network latency, a common bottleneck in cloud-based applications.
Enhances data privacy and security, as sensitive data remains on the device and isn't sent over networks where it could be intercepted.
Complies with regulations like GDPR that govern how user data can be stored and transferred.
Edge computing is more resistant to privacy breaches, although it doesn't eliminate them entirely.
To implement edge computing, edge devices must have sufficient computational power, memory to store ML models, and a reliable energy source. Deploying large models on underpowered devices can quickly deplete their batteries.

Many companies are competing to develop optimized edge devices for various ML use cases. Tech giants like Google, Apple, and Tesla have announced plans to create their own specialized chips, while ML hardware startups have raised substantial funding for AI chip development. It is predicted that by 2025, the number of active edge devices worldwide will exceed 30 billion.

The challenge is efficiently running ML models on a variety of hardware. The following section discusses how to compile and optimize models for specific hardware backends and introduces important concepts related to managing models at the edge, including intermediate representations (IRs) and compilers.

Compiling and Optimizing Models for Edge Devices:

Running a model built with a particular framework (e.g., TensorFlow, PyTorch) on a hardware backend requires support from the hardware vendor.
Offering support for a framework on a hardware backend involves time-intensive engineering work to map ML workloads to the hardware's design.
Different hardware backends have distinct memory layouts and compute primitives, necessitating tailored optimization.
Developing middlemen known as intermediate representations (IRs) bridges the gap between frameworks and platforms. Framework developers translate their code into this intermediary, simplifying support for different hardware.
Model optimization involves both local and global approaches to improve the model's efficiency, including techniques like vectorization, parallelization, loop tiling, and operator fusion.
These optimizations, when applied across multiple frameworks, can lead to substantial performance gains.
ML-powered compilers use machine learning to fine-tune and optimize models for specific hardware backends. These compilers can automatically optimize models and avoid the need for manual optimization engineers.

ML in Browsers:

WebAssembly (WASM) is a promising technology to run ML models in browsers, expanding compatibility across various devices.
WASM allows you to compile your models into an executable format for web-based applications.
While WASM is relatively faster than JavaScript, it's still slower than native code execution on devices.
The support for WASM is widespread, with 93% of devices worldwide supporting it as of September 2021


