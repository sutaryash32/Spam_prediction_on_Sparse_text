# Spam_prediction_on_Sparse_text

The first row lists features like "the", "to", "ect", etc., and each email row has counts of these words. The last column is "Prediction", which I assume is the class label (spam or not spam). The data seems to have word frequencies for each email.
First,  split the data into training and testing sets apply Naive Bayes directly.Each email has counts for each word. The first column is "Email No.", which is an identifier. The last column is "Prediction", which is the target variable (probably 0 for not spam, 1 for spam or similar).
looking at the emails, like Email 1 has Prediction 0, Email 2 has 1, etc. So the Prediction is the class label.

First, calculate the prior probabilities: P(Prediction=0) and P(Prediction=1) based on the frequency of each class in the training data.

Then, for each word, calculate the likelihood P(word|Prediction=0) and P(word|Prediction=1). This is done by counting the total occurrences of each word in each class, adding a smoothing term (Laplace smoothing) to avoid zero probabilities.

Given that the data has 8 emails, maybe with some 0s and 1s in Prediction. Wait, looking at the data:

Email 1: Prediction 0

Email 2: Prediction 1

Email 3: Prediction 0

Email 4: Prediction 0

Email 5: Prediction 0

Email 6: Prediction 1

Email 7: Prediction 0

Email 8: Prediction 0

So total 8 emails, 2 are class 1 (spam?), 6 are class 0.

Priors:

P(0) = 6/8 = 0.75

P(1) = 2/8 = 0.25

Then, for each word, compute the likelihoods.

Take the word "the". For class 0, sum all "the" counts in emails where Prediction=0. Let's compute:

Email 1: "the" count=0 → 0

Email 3: "the"=0

Email4: "the"=0

Email5: "the"=7

Email7: "the"=5

Email8: "the"=0

So total "the" in class 0: 0+0+0+7+5+0=12

Similarly for class 1: Emails 2 and 6.

Email2: "the"=8

Email6: "the"=4

Total "the" in class 1: 8+4=12

Total words in class 0: sum all word counts in all class 0 emails, in Multinomial NB, we calculate for each word, the count in each class, add Laplace smoothing (alpha=1 usually), then divide by total count + alpha * number of features.


Key Components:

1)Data Handling:

->Drops irrelevant columns (Email No.)

->Uses all word counts as features

->Prediction column as target

2)Model Training:

->Uses MultinomialNB (suitable for discrete counts)

->alpha=1 applies Laplace smoothing to handle zero probabilities

3)Evaluation:

->Accuracy score

->Confusion matrix
