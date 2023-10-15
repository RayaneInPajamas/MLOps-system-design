# Chapter 5. Feature Engineering
After publishing the paper “Practical Lessons from Predicting Clicks on Ads at Facebook” which claimed that having the right features is the most important thing in developing their ML models --> having the right feature tends to give them the biggest performance boost compared to clever algorithmic techniques such as hyperparameter tuning.

## Learned Features Versus Engineered Features
question : “Why do we have to worry about feature engineering? Doesn’t deep learning promise us that we no longer have to engineer features?”
Answer :sometimes this is right For this reason, deep learning is sometimes called feature learning.Many features can be automatically learned and extracted by algorithms(End to end deep learning) .But, the majority of ML applications in production aren’t deep learning.

### what features can be automatically extracted vs what features still need to be handcrafted
Sentiment analysis to classify whether a comment is spam or not
Before deep learning :when given a piece of text :
* have to manually apply classical text processing techniques such as lemmatization, expanding contractions, removing punctuation, and lowercasing everything.
* split the text into n-grams with n values of your choice.
<img src="05-Feature Engineering/01.png">

Feature engineering requires knowledge of domain-specific techniques—in this case, the domain is natural language processing (NLP) and the native language of the text

Much of this pain has been alleviated since the rise of deep learning. Instead of having to worry about lemmatization, punctuation, or stopword removal, you can just split your raw text into words (i.e., tokenization), create a vocabulary out of those words, and convert each of your words into one-shot vectors using this vocabulary.

Similar progress has been made for images too. Instead of having to manually extract features from raw images and input those features into your ML models, you can just input raw images directly into your deep learning models.

In ML systems:
system will likely need data beyond just text and images. For example, when detecting whether a comment is spam or not, on top of the text in the comment itself, you might want to use other information about:
*The comment*
How many upvotes/downvotes does it have?

*The user who posted this comment*
When was this account created, how often do they post, and how many upvotes/downvotes do they have?

*The thread in which the comment was posted*
How many views does it have? Popular threads tend to attract more spam.

The process of **choosing what information** to use and how to extract this information into a format usable by your ML models is feature engineering.For complex tasks such as recommending videos for users to watch next on TikTok, the number of features used can go up to millions. For domain-specific tasks such as predicting whether a transaction is fraudulent, you might need subject matter expertise with banking and frauds to be able to come up with useful features.

## Common Feature Engineering Operations
They include handling missing values, scaling, discretization, encoding categorical features, and cross features as well as the newer and exciting positional features.

### Handling Missing Values
Not all types of missing values are equal 
Example of house pricing prediction:
<img src="05-Feature Engineering/02.png">

There are three types of missing values:
*Missing not at random (MNAR)*
The income values are missing for reasons related to the values themselves.
*Missing at random (MAR)*
This is when the reason a value is missing is not due to the value itself, but due to another observed variable. In this example, we might notice that age values are often missing for respondents of the gender “A,” which might be because the people of gender A in this survey don’t like disclosing their age.
*Missing completely at random (MCAR)*
This is when there’s no pattern in when the value is missing. In this example, we might think that the missing values for the column “Job” might be completely random, not because of the job itself and not because of any other variable. People just forget to fill in that value sometimes for no particular reason. However, this type of missing is very rare. There are usually reasons why certain values are missing, and you should investigate.

**Deletion**
not a better method, but because it’s easier to do.
- column deletion : variable has too many missing values,
- row deletion : if a sample has missing value(s), just remove that sample. (MCAR) and number of examples with missing values is small, such as less than 0.1%.However, removing rows of data can also remove important information that your model needs to make predictions, especially if the missing values are not at random (MNAR).

**Imputation**
“fill the missing values with certain values.” Deciding which “certain values” to use is the hard part.
- default value "" for example --> problems in training (0 age)
Risks:
- adding noise to data
- data leakage

### Scaling
House pricing : annual incomes 45k-150k and number of rooms is 10 --> features are seen as numbers for the model and don't have a meaning it’s important to scale them to be similar ranges.Neglecting to do so can cause your model to make gibberish predictions, especially with classical algorithms like gradient-boosted trees and logistic regression.

*Ways of scaling features:*
An intuitive way to scale your features is to get them to be in the range [0, 1]. Given a variable x, its values can be rescaled to be in this range using the following formula:
<img src="05-Feature Engineering/03.png">

If you want your feature to be in an arbitrary range [a, b]—empirically, I find the range [–1, 1] to work better than the range [0, 1]—you can use the following formula:
<img src="05-Feature Engineering/04.png">

Scaling to an arbitrary range works well when you don’t want to make any assumptions about your variables. If you think that your variables might follow a normal distribution, it might be helpful to normalize them so that they have zero mean and unit variance. This process is called standardization:
<img src="05-Feature Engineering/05.png">

In practice, ML models tend to struggle with features that follow a skewed distribution. To help mitigate the skewness, a technique commonly used is log transformation: apply the log function to your feature.
<img src="05-Feature Engineering/06.png">


### Discretization
Discretization is the process of turning a continuous feature into a discrete feature.
Example:
- Lower income: less than $35,000/year
- Middle income: between $35,000 and $100,000/year
- Upper income: more than $100,000/year
Instead of having to learn an infinite number of possible incomes,our model can focus on learning only three categories, which is a much easier task to learn.
Age example
- Less than 18
- Between 18 and 22
- Between 22 and 30
- Between 30 and 40
- Between 40 and 65
- Over 65

Downside : 
the category boundary : $34,999 is now treated as completely different from $35,000, which is treated the same as $100,000.

### Encoding Categorical Features
People who haven’t worked with data in production tend to assume that categories are static, which means the categories don’t change over time. True for many categories:age brackets and income brackets are unlikely to change.

In production, categories change : product brand when used in poduction : example: Amazon : 2million brand in 2019 and the number is increasing.
- model crashes when encountring a new product : add Unkown category
- sellers complain that their new product are not getting any traffics : encoding 99% of the product of most popular brand
- new brands joined your site; some of them are new luxury brands, some of them are sketchy knockoff brands, some of them are established brands. They are treated as unkown

How to solve this kind of problems : 
*hashing trick* : The gist of this trick is that you use a hash function to generate a hashed value of each category. hashed value will become the index of that category.

### Feature Crossing
Feature crossing is the technique to combine two or more features to generate new features. feature crossing helps model nonlinear relationships between variables
<img src="05-Feature Engineering/07.png">

### Discrete and Continuous Positional Embeddings
An embedding is a vector that represents a piece of data. We call the set of all possible embeddings generated by the same algorithm for a type of data “an embedding space.” All embedding vectors in the same space are of the same size.

embedding has become a standard data engineering technique for many applications in both computer vision and NLP.

Example : language modeling where you want to predict the next token based on the previous sequence of tokens. (word embedding and position embedding [for transformers])
<img src="05-Feature Engineering/08.png">
Position embeddings can also be fixed.
<img src="05-Feature Engineering/09.png">


## Data Leakage
Data leakage refers to the phenomenon when a form of the label “leaks” into the set of features used for making predictions, and this same information is not available during inference. The leakage is nonobvious it can cause your models to fail in an unexpected and spectacular way

cancer diagnosis from images : Hospital A =/= Hospital B data distribution different
### Common Causes for Data Leakage
#### Splitting time-correlated data randomly instead of by time
Randomly splitting time-correlated data into training, validation, and test sets can lead to data leakage. Time-correlated data means that the timing of data generation affects its labels. For example, stock prices tend to move together. To avoid this, split your data by time, not randomly.


#### Scaling before splitting
Scaling data (e.g., normalizing) should be done after splitting the data. Using global statistics calculated on the entire dataset before splitting can leak information from the test set into the training process
#### Poor handling of data duplication before splitting
When filling missing data with statistics (e.g., mean or median), use only the statistics from the training set. Avoid using statistics calculated on the entire dataset, which can cause leakage.

#### Data duplication
Duplicate or near-duplicate data should be removed before splitting. Otherwise, the same samples may appear in both the training and validation/test sets

#### Group leakage
When groups of examples with correlated labels are divided into different splits, this can cause leakage. For example, if CT scans of the same patient are in different splits. Understanding how your data was generated is crucial to avoid this

#### Leakage from data generation process
Sometimes, data leakage occurs due to the data generation process. Understanding how data is collected and processed is essential to mitigate this type of leakage

### Detecting Data Leakage

#### Monitor Predictive Power
Measure the correlation between each feature and the target variable. Investigate features with unusually high correlations, as they may indicate potential leakage
#### Ablation Studies
Conduct ablation studies to measure the importance of features. If removing a feature significantly impacts model performance, investigate its role and potential for leakage
#### Watch for New Features
Be cautious when adding new features to your model. If a new feature substantially improves performance, it could indicate the presence of leaked information
#### Test Split Usage
Only use the test split for reporting the final model performance. Avoid using it for model development or generating new ideas

## Engineering Good Features
Why Good Feature Engineering Matters?
- More features don't always mean better model performance. Too many features can lead to issues like data leakage, overfitting, increased memory requirements, and longer inference times.
- Unused features become technical debts, and changes in your data pipeline may require updates to affected features.
- Removing unnecessary features can help models learn faster and improve performance.
- You can store removed features or feature definitions for reusability and sharing across teams

## Evaluating Feature Quality:

### Feature Importance

- Use methods like XGBoost's feature importance or SHAP (SHapley Additive exPlanations) to measure how much a feature contributes to a model's performance.
- Feature importance can help in choosing the right features and understanding model behavior.
- Often, a small number of features are responsible for a large portion of a model's feature importance
<img src="05-Feature Engineering/10.png">
<img src="05-Feature Engineering/11.png">

### Feature Generalization

- Features should generalize well to unseen data, as the goal of machine learning is to make accurate predictions on new data.
- Consider two aspects of generalization: feature coverage and distribution of feature values.
- Coverage is the percentage of samples with values for a feature. Features with very low coverage might not be useful.
- Carefully analyze how features' values are distributed in both seen (train) and unseen (test) data. Mismatches can harm model performance.
- Consider trade-offs between generalization and specificity when creating features. Features like "IS_RUSH_HOUR" can be more generalizable but less specific than "HOUR_OF_THE_DAY."
