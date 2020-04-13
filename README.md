# Covid19
#Introduction
Covid-19 is a global outbreak, started in Wuhan, China and not its everywhere. Its symptoms are very common and does not show a clear pattern, it will be more easier to act for the government if we have the data of the people with symptoms and can predict if they are infected or not and then the action can be taken accordingly in time. 
# What it does
This Machine Learning project is a predictive model that predicts if a particular person is infected or not with the coronavirus disease, given some of the information of the people that includes the data of their symptoms such as - Fever, Body Pain, Age, Runny nose, Breathing Problem, Infected.
# How I built it
For building this project, I have used libraries such as - numpy, pandas, matplotlib, seaborn and sklearn. I have built the project on jupyter notebook using a data of shape-[2000,6], in which their are 6 variables with one target variable - ['Infected'] and five features - ['Fever', 'BodyPain', 'Age', 'Runnynose', 'BreathingProblem']. Then I built split that data into train and test data and applied five algorithms on the top of it and they are - logistic regression, kNN, random forest, decision tree and naive bayes. My motive is to find the best algorithm that works best on the given data or shows the highest accuracy in predicting the target variable.
# Challenges I ran into
In this project, I have used different algorithms and they all have some different metrics to measure their performance like confusion matrix, R-square, MAPE, accuracy score etc. But I have to compare their performance using only one metric, so I choose accuracy score, to know each of their accuracy and compare their predictions. And the data available is not good as other relevant information are not given such as their geographical location, travel history, current stage of the country etc., then I can build a more efficient model.
# Accomplishment/Result
I was successful in building a project using five different machine learning algorithms. And found that Random Forest model gives the maximum accuracy among all of the models.
# Feedback
Though Random Forest Model gives the maximum accuracy, but still it is not enough and that can be because of the incomplete data or information.
