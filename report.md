# Evaluating loan risk at a peer to peer lending company
2
​
3
## Overview
4
The purpose is to build a model that can identify the creditworthiness of borrowers at a peer to peer lending company.
5
​
6
I am using a set of  roughly `77.000` historical  loans that each have 7 feautures and an outcome of either a healthy loan or a high risk loan.
7
The features are:
8
​
9
* loan size
10
* interest rate
11
* borrower income
12
* debt to income ratio
13
* number of accounts'
14
* derogatory marks
15
* total debt
16
​
17
The goal is to create a model with which I can accurately predict if a loan is healthy or at high risk of defaulting.
18
 
19
In analysing the dataset a concern is the low number (3%) of high risk loans on which a prediction can be built.
20
 
21
### Logistic regression
22
 
23
 I started creating a Logistic Regression model that I trained and tested using the historical data.
24
 
25
 ```python
26
 confusion_matrix(y_test, predictions)
27
 ```
28
 ![Screenshot 2023-04-15 at 3 44 06 pm](https://user-images.githubusercontent.com/112833174/232186928-e8d8af03-203c-47fa-b57d-c918abc6b331.png)
29
 
30
In the above table I have highlighted the main problem.
31
In these <mark>56</mark> cases the model would have advised a loan was healthy  where in 'reality' the loan was at high risk in defaulting and resulting in the lender not having the loan and it's interest repaid and thus loosing money.  
32
 
33
 ```python
34
print(classification_report(y_test, predictions,  
35
target_names=["healthy loan", "high risk loan"])) 
36
 ```
37
![Screenshot 2023-04-15 at 3 44 29 pm](https://user-images.githubusercontent.com/112833174/232186940-9ed1e120-d753-4dad-ad83-101c1933ae77.png)
38
​
39
Overall the report looks healthy but again when we scrutnise the high risk loans we find several prolem areas. Only 85% of the predicted  high risk loans were correct.  More importantly the recall rate on high risk loans is only 91% and efforts could, in my eyes should, be made to increase the recall rate of high risk loans from 91% to closer to 100% using another model.
40
​
41
​
42
### Logistic Regression with RandomOverSampler
43
​
44
I used a second model adding the imbalanced-learn library to increase the number of instances of high risk loans in the algorithm. In my model I upped the number high risk loans to that of the number of healthy loans. 
45
​
46
```python
47
oversample = RandomOverSampler(sampling_strategy='minority', 
48
                         random_state=1)
