# Evaluating loan risk at a peer to peer lending company

## Overview
The purpose is to build a model that can identify the creditworthiness of borrowers at a peer to peer lending company.

I am using a set of  roughly `77.000` historical  loans that each have 7 feautures and an outcome of either a healthy loan or a high risk loan.
The features are:

* Loan size
* interest rate
* borrower income
* debt to income ratio
* number of accounts'
* derogatory marks
* total debt

The goal is to create a model with which I can accurately predict of a loan is healthy or at high risk of defaulting.
 
In analysing the dataset a concern is the low number (3%) of high risk loans on which I can build a prediction.
 
### Logistic regression
 
 I started creating a Logistic Regression model that I trained and tested using the historical data.
 
 ```python
 confusion_matrix(y_test, predictions)
 ```
 ![Screenshot 2023-04-15 at 3 44 06 pm](https://user-images.githubusercontent.com/112833174/232186928-e8d8af03-203c-47fa-b57d-c918abc6b331.png)
 
In the above table I have highlighted the main problem.
In these <mark>56</mark> cases the model would have advised a loan was healthy  where in 'reality' the loan was at high risk in defaulting and resulting in the lender not having the loan and it's interest repaid and thus loosing money.  
 
 ```python
print(classification_report(y_test, predictions,  
target_names=["healthy loan", "high risk loan"])) 
 ```
![Screenshot 2023-04-15 at 3 44 29 pm](https://user-images.githubusercontent.com/112833174/232186940-9ed1e120-d753-4dad-ad83-101c1933ae77.png)

Overall the report looks healthy but again when we scrutnise the high risk loans we find several prolem areas. Only 85% of the predicted  high risk loans were correct.  More importantly the recall rate on high risk loans is only 91% and efforts could, in my eyes should, be made to increase the recall rate of high risk loans from 91% to closer to 100% using another model.


### Logistic Regression with RandomOverSampler

I used a second model adding the imbalanced-learn library to increase the number of instances of high risk loans in the algorithm. In my model I upped the number high risk loans to that of the number of healthy loans. 

```python
oversample = RandomOverSampler(sampling_strategy='minority', 
			 random_state=1)
```
![Screenshot 2023-04-15 at 3 44 45 pm](https://user-images.githubusercontent.com/112833174/232186968-056791c3-e71d-46ff-87f7-edd6ead85bb6.png)
![Screenshot 2023-04-15 at 3 45 01 pm](https://user-images.githubusercontent.com/112833174/232186974-00db605b-40a5-428a-938d-76c85cae2603.png)

Despite the precision dipping slightly to 84% and an increase in predictions that a loan was high risk when in 'reality' it was healthy this model is superior to the previous standard Logistic Regression model.  

Using the OverSampler module  only <mark>4</mark> loans were predicted to be healthy but in 'reality' were high risk.As a result the recall rate on high risk loans improved drastically from 91% to 99%. 

   
## Summary conclusions

Using a Logistic Regression model complemented with an imbalance-learn library is clearly preferred as a model to predict high risk loans at this peer to peer loan business.

Only 1% rather than 9% of the high risk loans are classified as healthy loans thus reducing the chance a lender would loose money.

It is however importamty to recognise this model is using historical data and does not take any external fluctuations such as for example; recession, climate change or advances in technology. It is also unknown which 'sectors' are (under)represented within the portfolio of this peer to peer loan business.

I also recommend that if you want to reduce the risks of loosing money you spread the amount 'invested' over more than 1 loan and even over as many loans as possible whilst realising that such a strategy would possibly reduce the average rate of return.

 
