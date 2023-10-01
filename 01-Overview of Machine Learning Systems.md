# Overview of Machine Learning Systems

## Introduction:
Incorporation of  multilingual neural machine translation system into google translate (first success of deep artificial neural networks in the production at scale) ---> translation improovement (more than 10 years vombined)

more companies were interested in including machine learning as a solution for their  problems. ML inflenced every field : communication, working, finding love, healthcare, transportation, farming and helping human understand th univers ... 

**false :** machine learning system is an ML algorithm used such as logistic regression.
**true :** machine learning algorithm is a small part of ML system in production . System includes : business reqirement,interface (where dev & users could interact with the system),data stack , the logic for developing, monitoring and updating models and the infrastructure that enables the delivery of that logic.

## relationship between MLOPS and ML system design:
**MLOps (machine learning  operations):** set of tools & best practices to bring machine learning in production.
**ML system design :** system approach to MLOps, which means that it considers an ML system to ensure that all compenent components & their stakeholders can work together to satisfy the objective & requirement 

## When to use machine learning 
Machine learning is not a tool to solve all the problems, even if it does it might not be the optimal solution.

two question to be asked : 
- is it necessary
- is it cost effective

**definition of what ML solutions do:**
Machine learning is an approach to : *learn* *complex pattern* from *existing data* and use these patterns to make *prediction on* unseen data 

- Learn: the system has the capacity to learn :
Learn is not just about stocking raw data or stating explicitly the relationship into  entities with no capacity of auto update (so relational database is not one : it cannot find out therelationship between two columns by itself).

ML systems learn from data, in supervised learning : based on example input and output pairs, example : build an ML system to learn to predict the rental price for houses at immotify startup, need to take input listing with relevent characteristics (square footage, number of rooms, neighborhood,cities ...) associated output is the house price & once learnt the system should be able to predict the price of a new house based on the given caracteristics.

- Complex patterns: there are patterns to learn, and they are complex :
ML solutions are only useful when there are patterns to learn meaning there is a model that can approximatly represent distribution of the data & with that model & that model could go and predict the future (unlike the fair dice, stock market has a certain pattern that could be deduced based on data). But even if the pattern is obvious or exists the dataset or ML algorithm could not be sufficient to capture it (Elon musk tweet for example could affect cryptocurrency prices).Even if all models fail to make reasonable predictions of cryptocurrency prices, it doesn’t mean there’s no pattern.

ML system could be useful in noticing what is to human a complex pattern (relationship between a house price & its characteristics) or simple pattern (recognizing object detection & speech recognition [argualbly]). What is complex to machines is different from what is complex to humans. 

ML is also called Software 2.0 because in traditional software we provided the function that calculate the prediction but in ML we provided the data & it provided back the function.
<img src="01-Overview of Machine Learning Systems/01.png">

- Existing data: data is available, or it’s possible to collect data :
The existence of data is crucial for deducing a pattern in order to make the prediction.

**Other approach :** *zero shot learning* where ML system makes predictions for a
task without having been trained on data for that task. However, this ML system was *previously trained on data for other tasks*, often *related* to the task in consideration.

**Other approach :** context of continual learning, deployement of ML system without them being trained on any data but it'll eventually (& hopefully learn) from incoming data in production . As a *drawback* it comes with poor user experience.

**Other approach :** Without data and without continual learning, many companies follow a “fake-it-til-you make it” approach: launching a product that serves predictions made by humans, instead of ML models, with the hope of using the generated data to train ML models later.

- Predictions: it’s a predictive problem : 
ML models make predictions, so they can only solve problems that require predictive answers (estimate a value in the future), since it is effective many problems were reframed as predictive problems as *“What would the answer to this question be?”*.
Compute-intensive problems are one class of problems that have been very successfully reframed as predictive, Instead of computing the exact outcome of a process (very expensive simulation) problem is reframed as “What would the outcome of this process look like?”, and approximate it using an ML model.

- Unseen data: unseen data shares patterns with the training data :
But this comes in the condition that the unseen data comes with the sam distribution of the seen data (security camera have not the same quality as phone cameras & thus not the same distribution)

- It’s repetitive : 
ML algorithms require a LOT of data to learn a pattern & since the process of training is repetetive each pattern is repeated multiple times, which makes it easier for machines to learn it.

- The cost of wrong predictions is cheap :
Unless your ML model’s performance is 100% all the time, which is highly unlikely for any meaningful tasks, your model is going to make mistakes. Even it gets the prediction wrong, the biggest case of use of ML is helping to making decision so in the end the last one to make the decision is the human (example of clicking at a video that is recommended although it shouldn't).
If one prediction mistake can have catastrophic consequences, ML might still be a suitable solution if, on average, the benefits of correctpredictions outweigh the cost of wrong predictions. (example of self driving cars)

- It’s at scale
the system makes a lot of predictions (generally series of predictions even if some tasks seemed easy & simple it is usually decomposed into smaller task (sometimes pipeline), and it's continuous --example of presidential election prediction--)

- The patterns are constantly changing
example of spam : spam email is a Nigerian prince, but tomorrow it might be a distraught Vietnamese writer.
Good news : the data changes slowly, so ML model can be updated with new data without having to figure out how the data has change & so the system can adapt to the changing data distribution using *continual learning*.


## Restrictions about usage of ML
ML algorithm shouldn't be used if & only if :
- It’s unethical
- There is simpler solutions
- Not cost-effective

If ML can't solve the entire problem, the problem could be broke down into smaller component where some of them could be solved using ML.

"if you can’t build a chatbot to answer all your customers’ queries, it might be possible to build an ML model to predict whether a query matches one of the frequently asked questions."


## Machine Learning Use Cases
With explosion of information & services ML manifested in either *search engine* or in *recommender system*. In smartphone ML has been used to predict the next thing user would type based on the beginning of the sentense knowed as *predictive typing*, *enhancing* the photo's quality, *fingerprint* on the face of the user. ML is also used in *machine translation*, *speech recognition* in Alexa & google Assistant, smart security cameras where cameras can detect agression act, or some uninvited guest...
In most cases enterprise applications might have stricter accuracy requirements but be more forgiving with latency requirements. For example, improving a speech recognition system’s accuracy from 95% to 95.5% might not be noticeable to most consumers, but improving a resource allocation system’s efficiency by just 0.1% can help a corporation like Google or General Motors save millions of dollars.
On the other hand,latency of a second might get a consumer distracted and opening something else, but enterprise users might be more tolerant of high latency.

## 2020 state of enterprise machine learning. 
<img src="01-Overview of Machine Learning Systems/02.png">
- Fraud detection is among the oldest applications of ML in the enterprise world.
- Deciding how much to charge for your product or service
-Price optimization is the process of estimating a price at a certain time period to maximize a defined objective function, such as the company’s margin, revenue, or growth rate. (doing optimization problems with machine learning)
- Churn prediction is predicting when a specific customer is about to stop using your products or services so that you can take appropriate actions to win them back. (studying the behaviour of the customer)
- Brand monitoring : using sentiment analysis to choose a brand's name or to design a logo ...
- Healthcare : detecting skin cancer or the state of progression of diabetes based on their ey pictures

## Understanding Machine Learning Systems
Is helpful to design & develop them 
### Machine Learning in Research Versus in Production

|                | Research                | Production                  |
|----------------|-------------------------|-----------------------------|
| Requirements   | State-of-the-art model  | Different stakeholders have |
|                | performance on benchmark | different requirements      |
|----------------|-------------------------|-----------------------------|
| Computational  | Fast training, high      | Fast inference, low latency |
| priority       | throughput               |                             |
|----------------|-------------------------|-----------------------------|
| Data           | Static                  | Constantly shifting         |
|----------------|-------------------------|-----------------------------|
| Fairness       | Often not a focus       | Must be considered          |
|----------------|-------------------------|-----------------------------|
| Interpretability | Often not a focus     | Must be considered          |
|----------------|-------------------------|-----------------------------|


*ML Fairness* is the model's quality or state of being fair or impartial, and it relates to the harm of allocation and quality of services

### Different stakeholders and requirements
Project usecase : Consider a mobile app that recommends restaurants to users. The app
makes money by charging restaurants a 10% service fee on each order. <=> expensive orders give the app more money than cheap orders.
The project involves ML engineers, salespeople, product managers, infrastructure engineers, and a manager:
#### ML engineers
Want a model that recommends restaurants that users will **most likely** order from : more complex model & more data

#### Sales team
Wants a model that recommends the **more expensive restaurants** since these restaurants bring in more service fees.

#### Product team
Notices that every increase in latency leads to a drop in orders throughthe service, so they want a model that can return the recommended restaurants in less than **100 milliseconds** so basically they recommend the *threshold metrics*.

#### ML platform team
Think about solutions for problem scaling as the application grows

#### Manager
Wants to maximize the margin, and one way to achieve this might be to let go of the ML team.

In this usecase we are exposed to a *decoupling objectives* problem 
We need to choose between several models in order to satisfy all the stakeholders
##### Example :
- Model A is the model that recommends the restaurants that users are most likely to click
- Model B is the model that recommends the restaurants that will bring in the most money for the app.

problem:they can’t return restaurant recommendations in less than 100 milliseconds.

Production team goal's to deploy the theory into practice : sometimes they cannot deploy ML techniques even if theoritically they perform very well (ensembling)

Do not underestimate small improvement in performence because it can result in a huge boost in revenue or cost savings. example : 0.2% improvement in the click-through rate for a product recommender system can result in millions of dollars increase in revenue for an ecommerce site.

### Computational priorities
- developing the model
- deploying the model 
- maintaining the model 
(last two are often)

In the process of developing a machine learning model, the most time-consuming phase is typically the training phase. This is because during training, the model learns from a large dataset by adjusting its parameters to make accurate predictions. Once the model has been deployed and is in production, its primary task is to generate predictions or make inferences based on new input data. At this stage, the inference process becomes the primary bottleneck, as it determines how quickly the model can process and provide predictions for real-time or batch applications.

To elaborate further:

Training Phase (Model Development): During this phase, the model is exposed to a substantial amount of training data, and it undergoes iterative optimization to improve its performance. This optimization often requires many passes through the training data, and it can be computationally intensive. Researchers and data scientists focus on developing techniques and architectures that allow for faster training times because reducing the training time expedites the model development process.

Inference Phase (Production): After a model is trained and deployed into a production environment, its primary role is to process incoming data and generate predictions or decisions in real-time or as needed. The speed and efficiency with which the model can perform these inference tasks become crucial for the overall system's performance and responsiveness. Production teams prioritize optimizing inference to ensure that predictions are generated quickly and efficiently, as slow inference can lead to delays in applications and decreased user satisfaction.

Problems with that : maintaining low latency in production (unlike in research)
One solution  : using batch queries (but this is also a solution that is taken in consideration in ML systems that runs in production)
<img src="01-Overview of Machine Learning Systems/03.png">

### Data
During the research phase, the datasets you work with are often *clean* and *well-formatted*, freeing you to focus on developing models. They are *static* by nature so that the community can use them to benchmark new architectures and techniques.
In production we are targeted by noisy, unstructured constantly shifting data (example of sequential data or real-time data collected by sensors & iot --> problem of transporting the data). In production the time we take into cleaning & structuring datasets is far bigger than this applied in research

### Fairness
Deployed model of the product must perform well and should NOT cause harm to the user because of miscalculation of the model .
"ML algorithms don’t predict the future, but encode the past, thus perpetuating the biases in the data and more."
When creating a model with 98% accuracy and there is 2% left (don't updat it) -- I guess this is why MLOps existed on the first place (to fill that 2%)
### Interpretability
Understanding how the model takes its decision will makes us answer the question : should we trust the Model ?
