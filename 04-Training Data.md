# Training data
it's about how to handle data from data science perspective

### problem with data 
Data is messy, complex, unpredictable, and potentially treacherous. it can easily sink the entire ML operation. But how to obtain / create good training data ? training, validation, and testing what are the major problems encountred (the lack of labels problem, the class imbalance problem, and techniques in data augmentation to address the lack of data problem)
difference between dataset & data is that dataset is finite & stationnary while data in production is neither finite nor stationary one of its main behaviour is (data distribution shifts)

## Sampling
- usually overlooked
- happens in many steps of an ML project lifecycle,
- Why sampling in probability theory :
    - When you don't have access to all world data (population) because it is too big you need to take a correct representator of it you take a sample of it & make the inference of the rest
    - When it when it’s infeasible to process all the data that you have access to, because it requires too much time or resources
    - Training on a subset of the data to see if the model is promising or not
- Why a chapter on sampling ?
    - avoid potential sampling biases
    - help us choose the methods that improve the efficiency of the data we sample.
- Families of probabilities :
    - nonprobability sampling
    - random sampling

### Nonprobability Sampling
the selection of data isn’t based on any probability criteria.
- Convenience sampling (we take the data available)
- Snowball sampling (example of tweeter account --> we moving to the next)
- Judgment sampling (Experts decide what samples to include)
- Quota sampling :selecting samples based on quotas for certain slices of data without any randomization

This type of sampling is quick but doesn't always create a reliable model

### Simple random sampling
you give all samples in the population equal probabilities of being selected (All categories are known).

pros :
- Easy to implement

cons :
- rare categories of data might not appear in your selection <-> model could think it is not an existing class.

### Stratified Sampling
it is about dividing the population into the groups that you care about and sample from each group separately. (to solve the problem of the first sampling)

cons :
- This is especially challenging when one sample might belong to multiple groups, as in the case of multilabel tasks. --> multilabled class

### Weighted Sampling
concept : each sample is given a weight, which determines the probability of it being selected. (A, B, and C, and want them to be selected with the probabilities of 50%, 30%, and 20% respectively,weights are 0.5,0.3,0.2)

When ?
leveraging the domain expertise if we know that a certain subpopulation of data, such as more recent data, is more valuable to the model

Weighted sampling is used to select samples to train your model with, whereas sample weights are used to assign “weights” or “importance” to training samples.

### Reservoir Sampling
assuming that we are sampling k samples of a "tweet population" the algorithm assures that :
- Every tweet has an equal probability of being selected
- You can stop the algorithm at any time and the tweets are sampled with the correct probability.

Concept :
0. reservoir (array)
1. Put the first k elements into the reservoir.
2. For each incoming nth element, generate a random number i such that 1 ≤ i ≤ n
3. If 1 ≤ i ≤ k: replace the ith element in the reservoir with the nth element. Else, do nothing.


### Importance Sampling
Importance sampling allows us to sample from a distribution when we only have access to another distribution.

- x from a distribution P(x)
- P(x) is really expensive, slow, or infeasible to sample from
- distribution Q(x) that is a lot easier to sample from.
- sample x from Q(x) instead and weigh this sample by [P(x)/Q(x)]

## Labeling
most ML models in production today are supervised,The performance of an ML model still depends heavily on the quality and quantity of the labeled data it’s trained on.
### Hand labels
- hand-labeling data can be expensive (require expertise)
- hand labeling poses a threat to data privacy
- hand labeling is slow (takes multple iterations)
#### Label multiplicity
relying on multiple source of annotator with different level of expertise (problem of label ambiguity or label multiplicity <--> different label for the same data)

#### Data lineage
examination of data quality is crutial because it can affect the performance of the model . because we must be aware of the distribution we are trying to approximate using the module

### Natural labels
Tasks with natural labels are tasks where the model’s predictions can be automatically evaluated or partially evaluated by the system. Example : model that estimates time of arrival for a certain route on Google Maps.

#### Feedback loop length
- Tasks with short feedback loops are tasks where labels are generally available within minutes.
- Tasks with long feedback loops are tasks where labels are generally available within hours (blog posts) or weeks (clothing recommendation)
- different types of feedback : for an e-commerce website clicking on a product recommendation, adding a product to cart, buying a product, rating, leaving a review, and returning a previously bought product.
### Handling the Lack of Labels
Methods :
- Weak supervision : Leverages (often noisy) heuristics to generate labels
- Semi-supervision : Leverages structural assumptions to generate labels
- Transfer learning : Leverages models pretrained on another task for your new task
- Active learning : Labels data samples that are most useful to your model
## Class imbalance
problem in classification tasks where there is a substantial difference in the number of samples in each class of the training data.
- insufficient signal for your model to learn to detect the minority classes.
- using simple heuristics (algorithm) could give very bad results 
### Handling Class Imbalance
- using very deep neural networks
- developing models is quite challenging
#### Using the right evaluation metrics
- maximizing metric & satifying metrics
- precision, recall, and F1
#### Data-level methods: Resampling
- Data-level methods modify the distribution of the training data to reduce the level of imbalance to make it easier for the model to learn.
- oversampling, adding more instances from the minority classes
- undersampling, removing instances of the majority classes.
#### Algorithm-level methods
If data-level methods mitigate the challenge of class imbalance by altering the distribution of your training data, algorithm-level methods keep the training data distribution intact but alter the algorithm to make it more robust to class imbalance.
## Data augmentation
- Data augmentation is a family of techniques that are used to increase the amount of training data.
- used for tasks that have limited training data, such as in medical imaging.

### Simple Label-Preserving Transformations

### Perturbation

### Data Synthesis