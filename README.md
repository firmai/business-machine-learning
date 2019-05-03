# Business Machine Learning and Data Science Applications 
[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/intent/tweet?text=A%20list%20of%20Python%20Notebooks%20for%20business%20applications&url=https://github.com/firmai/business-machine-learning&via=dereknow&hashtags=ML,AI,Python,business)

<p align="center">
  <img src="https://github.com/firmai/business-machine-learning/raw/master/assets/BML%402x-704x454.png">
</p>


A curated list of applied business machine learning (BML) and business data science (BDS) examples and libraries. The code in this repository is in Python (primarily using jupyter notebooks) unless otherwise stated. The catalogue is inspired by `awesome-machine-learning`.

***Caution:*** This is a work in progress, please contribute, especially if you are a subject expert in ML/DS for [Accounting](#accounting), [Banking, Finance and Insurance](#bankfin), [Customer](#customer), [Employee](#employee), [Legal](#legal), [Management](#management), [Operations](#operations) and [Public](#public) matters. 

If you want to contribute to this list (please do), send me a pull request or contact me [@dereknow](https://twitter.com/dereknow).
Also, a listed repository should be deprecated if:

* Repository's owner explicitly say that "this library is not maintained".
* Not committed for long time (2~3 years).

</br>
***If there is any contributors out there willing to help first populate and then maintain a Python analytics section in on of the following industries, please get in contact with me.***   
</br>


| Accommodation & Food             | Agriculture & Forestry           | Banking & Insurance               |
| -------------------------------- | -------------------------------- | --------------------------------- |
| Biotechnological & Life Sciences | Construction & Engineering       | Education & Research              |
| Emergency & Police               | Entertainment, Recreation & Arts | Goods & Manufacturing             |
| Government and Public Works      | Healthcare and Social Assitance  | Media & Publishing                |
| Mining, Oil & Gas Extraction     | Miscellaneous                    | Professional & Technical Services |
| Real Estate, Rental & Leasing    | Technology                       | Telecommunications                |
| Transportation & Warehousing     | Utilities                        | Wholesale & Retail                |
| Justice                          |                                  |                                   |




## Table of Contents

### Department Applications
<!-- MarkdownTOC depth=4 -->

- [Accounting](#accounting)
    - [Machine Learning](#accounting-ml)
    - [Analytics](#accounting-analytics)
    - [Textual Analysis](#accounting-text)
    - [Data](#accounting-data)
    - [Research and Articles](#accounting-ra)
    - [Websites](#accounting-web)
    - [Courses](#accounting-course)
- [Banking, Finance and Insurance](#bankfin)
    - [Consumer Financial](#bankfin-cf)
    - [Management and Operations](#bankfin-mo)
    - [Valuation](#bankfin-value)
    - [Fraud](#bankfin-fraud)
    - [Insurance and Risk](#bankfin-ir)
    - [Trading and Investments](#bankfin-ti)
    - [Physical](#bankfin-ph)
    - [Data](#bankfin-data)
- [Customer](#customer)
    - [Lifetime Value](#customer-clv)
    - [Segmentation](#customer-seg)
    - [Behaviour](#customer-behave)
    - [Recommender](#customer-rec)
    - [Churn Prediction](#customer-cp)
    - [Sentiment](#customer-sent)
    - [Demand Forcasting](#customer-df)
- [Employee](#employee)
    - [Management](#employee-man)
    - [Performance](#employee-perf)
    - [Turnover](#employee-general-turn)
    - [Conversations](#employee-con)
    - [Physical](#employee-ph)
- [Legal](#legal)
    - [Tools](#legal-tools)
    - [Policy and Regulatory](#legal-pr)
    - [Judicial](#legal-judicial)
- [Management](#management)
    - [Strategy](#management-strat)
    - [Decision Optimisation](#management-do)
    - [Causal Inference](#management-causal)
    - [Statistics](#management-stat)
    - [Quantitative](#management-quant)
    - [Data](#management-data)
- [Operations](#operations)
    - [Failures and Anomalies](#operations-fail)
    - [Load and Capacity Management](#operations-load)
    - [Prediction Management](#operations-predict)
- [Public](#public)
    - [Social Policies](#public-social)
    - [Election Analysis](#public-elect)
    - [Disaster Management](#public-dis)
    - [Urban Planning](#public-urban)
    - [Utilities](#public-util)


<!-- /MarkdownTOC -->

<a name="accounting"></a>
## Accounting

<a name="accounting-ml"></a>
#### Machine Learning
* [Chart of Account Prediction](https://github.com/agdgovsg/ml-coa-charging ) - Using labeled data to suggest the account name for every transaction.
* [Accounting Anomalies](https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab-solutions.ipynb) -  Using deep-learning frameworks to identify accounting anomalies.
* [Financial Statement Anomalies](https://github.com/rameshcalamur/fin-stmt-anom) - Detecting anomalies before filing, using R.
* [Useful Life Prediction (FirmAI)](http://www.firmai.org/documents/Aged%20Debtors/) - Predict the useful life of assets using sensor observations and feature engineering.
 
<a name="accounting-analytics"></a>
#### Analytics

* [Forensic Accounting](https://github.com/mschermann/forensic_accounting) - Collection of case studies on forensic accounting using data analysis.
* [General Ledger (FirmAI)](http://www.firmai.org/documents/General%20Ledger/) - Data processing over a general ledger as exported through an accounting system.
* [Bullet Graph (FirmAI)](http://www.firmai.org/documents/Bullet-Graph-Article/) - Bullet graph visualisation helpful for tracking sales, commission and other performance.
* [Aged Debtors (FirmAI)](http://www.firmai.org/documents/Aged%20Debtors/) - Example analysis to invetigate aged debtors.

<a name="accounting-text"></a>
#### Textual Analysis

* [Financial Sentiment Analysis](https://github.com/EricHe98/Financial-Statements-Text-Analysis) - Sentiment, distance and proportion analysis for trading signals.
* [Extensive NLP](https://github.com/TiesdeKok/Python_NLP_Tutorial/blob/master/NLP_Notebook.ipynb) - Comprehensive NLP techniques for accounting research.

<a name="accounting-data"></a>
#### Data, Parsing and APIs

* [EDGAR](https://github.com/TiesdeKok/UW_Python_Camp/blob/master/Materials/Session_5/EDGAR_walkthrough.ipynb) - A walk-through in how to obtain EDGAR data. 
* [IRS](http://social-metrics.org/sox/) - Acessing and parsing IRS filings.
* [Financial Corporate](http://raw.rutgers.edu/Corporate%20Financial%20Data.html) - Rutgers corporate financial datasets.
* [Non-financial Corporate](http://raw.rutgers.edu/Non-Financial%20Corporate%20Data.html) - Rutgers non-financial corporate dataset.
* [PDF Parsing](https://github.com/danshorstein/python4cpas/blob/master/03_parsing_pdf_files/AR%20Aging%20-%20working.ipynb) - Extracting useful data from PDF documents. 
* [PDF Tabel to Excel](https://github.com/danshorstein/ficpa_article) - How to output an excel file from a PDF.

<a name="accounting-ra"></a>
#### Research And Articles

* [Understanding Accounting Analytics](http://social-metrics.org/accountinganalytics/) - An article that tackles the importance of accounting analytics.
* [VLFeat](http://www.vlfeat.org/) - VLFeat is an open and portable library of computer vision algorithms, which has Matlab toolbox.

<a name="accounting-web"></a>
#### Websites

* [Rutgers Raw](http://raw.rutgers.edu/) - Good digital accounting research from Rutgers.

<a name="accounting-course"></a>
#### Courses

* [Computer Augmented Accounting](https://www.youtube.com/playlist?list=PLauepKFT6DK8TaNaq_SqZW4LIDJhCkZe2) - A video series from Rutgers University looking at the use of computation to improve accounting.
* [Accounting in a Digital Era](https://www.youtube.com/playlist?list=PLauepKFT6DK8_Xun584UQPPsg1grYkWw0) - Another series by Rutgers investigating the effects the digital age will have on accounting.

<a name="bankfin"></a>
## Banking, Finance and Insurance

<a name="bankfin-cv"></a>
#### Consumer Finance

* [Loan Acceptance](https://github.com/Paresh3189/Bankruptcy-Prediction-Growth-Modelling) - Classification and time-series analysis for loan acceptance. 
* [Predict Loan Repayment](https://github.com/Featuretools/predict-loan-repayment) - Predict whether a loan will be repaid using automated feature engineering.
* [Loan Eligibility Ranking](https://github.com/RealRadOne/Gyani-The-Loan-Eligibility-Predictor) - System to help the banks check if a customer is eligible for a given loan.
* [Home Credit Default (FirmAI)](http://www.firmai.org/documents/Aggregator/#each-time-step-takes-30-seconds) - Predict home credit default. 
* [Mortgage Analytics](https://github.com/abuchowdhury/Mortgage_Bank_Loan_Analtsics/blob/master/Mortgage%20Bank%20Loan%20Analytics.ipynb) - Extensive mortgage loan analytics.
* [Credit Approval](https://github.com/IBM-Cloud-DevFest-2018/Data-Science-for-Banking/blob/master/02-CreditCardApprovalModel/CreditCardApprovalModel.ipynb) - A system for credit card approval. 
* [Loan Risk](https://github.com/Brett777/Predict-Risk) - Predictive model to help to reduce charge-offs and losses of loans. 
* [Amortisation Schedule (FirmAI)](http://www.firmai.org/documents/Amortization%20Schedule/) - Simple amortisation schedule in python for personal use.  


<a name="bankfin-mo"></a>
#### Management and Operation

* [Credit Card](https://github.com/am-aditya/Artificial-Intelligence-for-Banking/blob/master/03_ipy_notebooks/clv_prediction.ipynb) - Estimate the CLV of credit card customers. 
* [Survival Analysis](https://github.com/am-aditya/Artificial-Intelligence-for-Banking/blob/master/01_code/01_02_clv_survival/Survival_Analysis.py) - Perform a survival analysis of customers.
* [Next Transaction](https://github.com/am-aditya/Artificial-Intelligence-for-Banking/blob/master/01_code/01_02_clv_survival/Customer_NextTransaction_Prediction.py) - Deep learning model to predict the transaction amount and days to next transaction. 
* [Credit Card Churn](https://github.com/am-aditya/Artificial-Intelligence-for-Banking/blob/master/01_code/01_02_clv_survival/Customer_NextTransaction_Prediction.py) - Predicting credit card customer churn.  
* [Bank of England  Minutes](https://github.com/sekhansen/mpc_minutes_demo/blob/master/information_retrieval.ipynb) - Textual analysis over bank minutes. 

<a name="bankfin-value"></a>
#### Valuation

* [Zillow Prediction](https://github.com/eswar3/Zillow-prediction-models) - Zillow valuation prediction as performed on Kaggle. 
* [Real Estate](https://github.com/denadai2/real-estate-neighborhood-prediction) - Predicting real estate prices from the urban environment.
* [Used Car](https://nbviewer.jupyter.org/github/albahnsen/PracticalMachineLearningClass/blob/master/exercises/P1-UsedVehiclePricePrediction.ipynb) - Used vehicle price prediction.


<a name="bankfin-fraud"></a>
#### Fraud

* [XGBoost](https://github.com/KSpiliop/Fraud_Detection) - Fraud Detection by tuning XGBoost hyper-parameters with Simulated Annealing
* [Fraud Detection Loan in R](https://github.com/longtng/frauddetectionproject/blob/master/A%20Consideration%20Point%20of%20%20Fraud%20Detection%20in%20Bank%20Loans%20Project%20Code.ipynb) - Fraud detection in bank loans.
* [AML Finance Due Diligence](https://github.com/Michaels72/AML-Due-Diligence/blob/master/AML_Finance_DD.ipynb) - Search news articles to do finance AML DD.
* [Credit Card Fraud](https://github.com/am-aditya/Artificial-Intelligence-for-Banking/blob/master/03_ipy_notebooks/fraud_detection.ipynb) - Detecting credit card fraud. 

<a name="bankfin-ir"></a>
#### Insurance and Risk

* [Bank Failure ](https://github.com/Shomona/Bank-Failure-Prediction/blob/master/Bank.ipynb) - Predicting bank failure. 
* [Risk Management](https://github.com/andrey-lukyanov/Risk-Management) - Finance risk engagement course resources. 
* [VaR GaN](https://github.com/hamaadshah/market_risk_gan_keras) - Estimate Value-at-Risk for market risk management using Keras and TensorFlow.
* [Actuarial Sciences (R)](https://github.com/JSchelldorfer/ActuarialDataScience) - A range of actuarial tools in R. 


<a name="bankfin-ti"></a>
#### Trading and Investment

* [Deep Portfolio](https://github.com/DLColumbia/DL_forFinance) - Deep learning for finance 
Predict volume of bonds.
* [AI Trading](https://github.com/borisbanushev/stockpredictionai/blob/master/readme2.md) - Modern AI trading techniques.
* [Corporate Bonds](https://github.com/ishank011/gs-quantify-bond-prediction) - Predicting the buying and selling volume of the corporate bonds.
* [Simulation](https://github.com/chenbowen184/Computational_Finance) - Investigating simulations as part of computational finance. 
* [Industry Clustering](https://github.com/SeanMcOwen/FinanceAndPython.com-ClusteringIndustries) - Project to cluster industries according to financial attributes.
* [Financial Modeling](https://github.com/MiyainNYC/Financial-Modeling/tree/master/codes) - HFT trading and implied volatility modeling. 
* [Trend Following](http://inseaddataanalytics.github.io/INSEADAnalytics/ExerciseSet2.html) - A futures trend following portfolio investment strategy. 
* [Financial Statement Sentiment](https://github.com/MAydogdu/TextualAnalysis) - Extracting sentiment from financial statements using neural networks.
* [Applied Corporate Finance](https://github.com/chenbowen184/Data_Science_in_Applied_Corporate_Finance) - Studies the empirical behaviors in stock market.
* [Market Crash Prediction](https://github.com/sarachmax/MarketCrashes_Prediction/blob/master/LPPL_Comparasion.ipynb) - Predicting market crashes using an LPPL model.
* [NLP Finance Papers](https://github.com/chenbowen184/Research_Documents_Curation_with_NLP) - Curating quantitative finance papers using machine learning.
* [ARIMA-LTSM Hybrid](https://github.com/imhgchoi/Corr_Prediction_ARIMA_LSTM_Hybrid) - Hybrid model to predict future price correlation coefficients of two assets
* [Basic Investments](https://github.com/SeanMcOwen/FinanceAndPython.com-Investments) - Basic investment tools in python. 
* [Basic Derivatives](https://github.com/SeanMcOwen/FinanceAndPython.com-Derivatives) - Basic forward contracts and hedging. 
* [Basic Finance](https://github.com/SeanMcOwen/FinanceAndPython.com-BasicFinance) - Source code notebooks basic finance applications. 


<a name="bankfin-ph"></a>
#### Physical

* [Bank Note Fraud Detection](https://github.com/apoorv-goel/Bank-Note-Authentication-Using-DNN-Tensorflow-Classifier-and-RandomForest) - Bank Note Authentication Using DNN Tensorflow Classifier and RandomForest.
* [ATM Surveillance](https://github.com/ShreyaGupta08/InfyHack) - ATM Surveillance in banks use case.

<a name="bankfin-data"></a>
#### Data

* [Employee Count SEC Filings](https://github.com/healthgradient/sec_employee_information_extraction) 
* [SEC Parsing](https://github.com/healthgradient/sec-doc-info-extraction/blob/master/classify_sections_containing_relevant_information.ipynb) 
* [Open Edgar](https://github.com/LexPredict/openedgar) 
* [Rating Industries](http://www.ratingshistory.info/) 


<a name="customer"></a>
## Customer

<a name="customer-clv"></a>
#### Lifetime Value
* [Pareto/NBD Model](https://github.com/datascienceinc/oreilly-intro-to-predictive-clv/blob/master/oreilly-an-intro-to-predictive-clv-tutorial.ipynb) - Calculate the CLV using a Pareto/NBD model.
* [Gamma-Gamma Model](https://github.com/GitiHubi/deepAI/blob/master/GTC_2018_Lab-solutions.ipynb) -  Using deep-learning frameworks to identify accounting anomalies.
* [Cohort Analysis](https://github.com/iris9112/Customer-Segmentation/blob/master/Chapter1-Cohort_Analysis.ipynb) - Cohort analysis to group customers into mutually exclusive cohorts measured over time. 

 
<a name="customer-seg"></a>
#### Segmentation

* [E-commerce](https://github.com/jalajthanaki/Customer_segmentation/blob/master/Cust_segmentation_online_retail.ipynb ) - E-commerce customer segmentation.
* [Groceries](https://github.com/harry329/CustomerFinding/blob/master/customer_segments.ipynb ) - Segmentation for grocery customers. 
* [Online Retailer](https://github.com/Vinayak02/CustomerCentricRetail/blob/master/CustomerSegmentation/Customer_Segmentation_Online_Retail.ipynb) - Online retailer segmentation.
* [Bank](https://github.com/Mogbo/Customer-Clustering-Segmentation-Challenge) - Bank customer segmentation.
* [Wholesale](https://github.com/SyedAdilAli93/Identifying-Customers/blob/master/customer_segments.ipynb) - Clustering of wholesale customers.
* [Various](https://github.com/abalaji-blr/CustomerSegments/tree/master/deliver ) - Multiple types of segmentation and clustering techniques. 


<a name="customer-behave"></a>
#### Behaviour

* [RNN](https://github.com/DaniSanchezSantolaya/RNN-customer-behavior/tree/master/src) - Investigating customer behaviour over time with sequential analysis using an RNN model.
* [Neural Net](https://github.com/Vinayak02/CustomerCentricRetail/blob/master/DemandForecasting/NeuralNetworks.ipynb) - Demand forecasting using artificial neural networks.
* [Temporal Analytics](https://github.com/riccotti/CustomerTemporalRegularities) - Investigating customer temporal regularities.
* [POS Analytics](https://github.com/IBM/customer_pos_analytics/blob/master/code/Customer%20Ranking%20POS%20wip.ipynb) - Analytics driven customer behaviour ranking for retail promotions using POS data.
* [Wholesale Customer](https://github.com/kralmachine/WholesaleCustomerAnalysis/blob/master/WhosaleCustomerAnalysis.ipynb) - Wholesale customer exploratory data analysis.
* [RFM](https://github.com/espin086/customer_growth/blob/master/rfm/rfm.ipynb) - Doing a RFM (recency, frequency, monetary) analysis. 
* [Returns Behaviour](https://github.com/adarsh2111/Customer-Returns-Analysis-Customer-Fraud-Detection-/blob/master/Returns%20Analysis.ipynb) - Predicting total returns and fraudulent returns. 
* [Visits](https://github.com/Ryanfras/Customer-Visits/blob/master/Customer%20Visits.ipynb) - Predicting which day of week a customer will visit.
* [Bank: Next Purchase](https://github.com/albertcdc/Project_CAJAMAR) - A project to predict bank customers' most probable next purchase.
* [Bank: Customer Prediction](https://github.com/rohangawade/Predicting-Target-customers-for-Bank-Policy-subscribtion-using-Logistic_Regression_Transparency) - Predicting Target customers who will subscribe the new policy of the bank.
* [Next Purchase](https://github.com/Featuretools/predict-next-purchase) - Predict a customers’ next purchase also using feature engineering. 
* [Customer Purchase Repeats](https://github.com/kpei/Customer-Analytics/blob/master/customer_zakka.ipynb) - Using the lifetimes python library and real jewellery retailer data analyse customer repeat purchases.
* [AB Testing](https://github.com/sushant2811/customerAnalyticsWithA-BTesting/blob/master/customerAnalyticsWithA-BTesting.ipynb) - Find the best KPI and do A/B testing.
* [Customer Survey (FirmAI)](http://www.firmai.org/documents/Customer%20Survey/) - Example of parsing and analysing a customer survey. 
* [Happiness](https://github.com/rohit6205/predictHappiness/blob/master/predictingHapiness.ipynb) - Analysing customer happiness from hotel stays using reviews. 
* [Miscellaneous Customer Analytics](https://github.com/mapr-demos/customer360) - Various tools and techniques for customer analysis. 


<a name="customer-rec"></a>
#### Recommender

* [Recommendation](https://github.com/annalucia1/Customer-Behavior-Analysis-Recommendation/blob/master/recomendation_by_RatingScore.ipynb) - Recommend the songs that a customer on a music app would prefer listening to. 
* [General Recommender](https://github.com/Vinayak02/CustomerCentricRetail/blob/master/RecommenderSystem/Recommender.ipynb) - Identifying which products to recommend to which customers. 
* [Collaborative Filtering](https://github.com/codeBehindMe/CustomerIntelligence/blob/master/CollaborativeFiltering.ipynb ) - Customer recommendation using collaborative filtering.
* [Up-selling (FirmAI)](http://www.firmai.org/documents/Expected%20Value%20Business%20Model%20Performance/ ) - Analysis to identify up-selling opportunities. 


<a name="customer-cp"></a>
#### Churn Prediction

* [Ride Sharing](https://github.com/MSopranoInTech/Churn-prediction/blob/master/Churn%20Prediction.ipynb) - Identify customer churn rates in order to target customers for retention campaigns. 
* [KKDBox I](https://github.com/naomifridman/Deep-VAE-prediction-of-churn-customer) - Variational deep autoencoder to predict churn customer
* [KKDBox II](https://github.com/Featuretools/predict-customer-churn) - A three step customer churn prediction framework using feature engineering. 
* [Personal Finance](https://github.com/smit5490/CustomerChurn) - Predict customer subscription churn for a personal finance business. 
* [ANN](https://github.com/AgarwalGeeks/customer-churn-Analysis/blob/master/ANN.ipynb) - Churn analysis using artificial neural networks. 
* [Bike](http://www.firmai.org/documents/Customer%20Segmentation/) - Customer bike churn analysis.
* [Cost Sensitive](https://nbviewer.jupyter.org/github/albahnsen/ML_RiskManagement/blob/master/exercises/10_CS_Churn.ipynb) - Cost sensitive churn analysis drivenby economic performance. 

<a name="customer-sent"></a>
#### Sentiment

* [Topic Modelling](https://github.com/Chrisjw42/ZLSurveyAnalysis) - Topic modelling on a corpus of customer surveys from the VR industry. 
* [Customer Satisfaction](https://github.com/BoulderDataScience/kaggle-santander) - Predict customer satisfaction using Kaggle data.

<a name="employee"></a>
## Employee

<a name="employee-man"></a>
#### Management
* [Personality Prediction ](https://github.com/jcl132/personality-prediction-from-text) - Predict Big 5 Personality from text. 
* [Salary Prediction Resume](https://github.com/Artifelse/Prediction-salary-on-the-base-of-the-resume/blob/master/NLP.ipynb) - Textual analyses over resume to predict appropriate salary.
* [Employee Review Analysis](https://github.com/jackyip1/Indeed-Reviews/blob/master/Python%20scripts/Indeed%20-%20Main.ipynb) - Review analytics for top 50 retail companies on Indeed.
* [Diversity Analysis](https://github.com/mtfaye/Employee-Diversity-in-Tech/blob/master/Data%20Viz%20Special%20Edition.ipynb) - A simple analysis of gender and race disparity in the tech industry.
* [Occupation Prediction](https://github.com/RashmiSingh24/OccuptionPrediction/blob/master/BurningGlass.ipynb) - Predict the likelihood that an occupation is analytical. 


<a name="employee-perf"></a>
#### Performance
* [Training Hours Performance](https://github.com/niqueerdo/MLpredictemployeedevelopment/blob/master/Working%20Ntbk_MODELS_Clustering.ipynb) - The impact of training ours on employee performance.
* [Promotion Prediction](https://github.com/AbinSingh/Employee-Promotion-Prediction/blob/master/Employee_Promotion_Prediction.ipynb) - Analysing promotion patterns. 
* [Employee Attendance prediction](https://github.com/lokesh1233/Employee_Attendance/tree/master/notebooks) - Various tools to predict employee attendance.


<a name="employee-turn"></a>
#### Turnover
* [Early Leaving Employees](https://github.com/anushuk/ML-Human-Resources-Analytics/blob/master/Human%20Resources%20Analytics.ipynb ) - Identifying why the best and most experienced employees leaving prematurely.
* [Employee Turnover](https://github.com/randylaosat/Predicting-Employee-Turnover-Complete-Guide-Analysis/blob/master/HR%20Analytics%20Employee%20Turnover/HR_Analytics_Report.ipynb) - Identifying factors associated with employee turnover.


<a name="employee-con"></a>
#### Conversations
* [Slack Communication Analysis](https://github.com/stiebels/slack_nlp/blob/master/Slack%20Analytics.ipynb) - Producing meaningful visualisations from slack conversations. 
* [Employee Relationships from Conversations ](https://github.com/yuwie10/cultivate) - Identifying employee relationships from emails for improved HR analytics.
* [Categorise Employee Requests](https://github.com/denizn/Request-classification-via-TFIDF) - Classifying employee requests via TFDIF Vectorizer and RandomForestClassifier.


<a name="employee-ph"></a>
#### Physical
* [Employee Face Recognition](https://github.com/ckarthic/Face-Recognition) - A face recognition implementation. 
* [Attendance Management System](https://github.com/mrsaicharan1/face-rec-a) - An attendance management system using face recognition.

<a name="legal"></a>
## Legal


<a name="legal-tools"></a>
#### Tools
* [LexPredict](https://github.com/LexPredict/lexpredict-contraxsuite ) - Software package and library. 
* [AI Para-legal](https://github.com/davidawad/lobe) - Lobe is the world's first AI paralegal.
* [Legal Entity Detection](https://github.com/hockeyjudson/Legal-Entity-Detection/blob/master/Dataset_conv.ipynb) - NER For Legal Documents.
* [Legal Case Summarisation](https://github.com/Law-AI/summarization) - Implementation of different summarisation algorithms applied to legal case judgements.
* [Legal Documents Google Scholar](https://github.com/GirrajMaheshwari/Web-scrapping-/blob/master/Google_scholar%2BExtract%2Bcase%2Bdocument.ipynb ) - Using Google scholar to extract cases programatically. 
* [Chat Bot](https://github.com/akarazeev/LegalTech) - Chat-bot and email notifications.


<a name="legal-pr"></a>
#### Policy and Regulatory
* [GDPR scores](https://github.com/erickjtorres/AI_LegalDoc_Hackathon) - Predicting GDPR Scores for Legal Documents.
* [Driving Factors FINRA](https://github.com/siddhantmaharana/text-analysis-on-FINRA-docs) - Identify the driving factors that influence the FINRA arbitration decisions.
* [Securities Bias Correction](https://github.com/davidsontheath/bias_corrected_estimators/blob/master/bias_corrected_estimators.ipynb ) - Bias-Corrected Estimation of Price Impact in Securities Litigation.
* [Public Firm to Legal Decision](https://github.com/anshu3769/FirmEmbeddings) - Embed public firms based on their reaction to legal decisions.


<a name="legal-judicial"></a>
#### Judicial Applied
* [Supreme Court Prediction](https://github.com/davidmasse/US-supreme-court-prediction) - Predicting the ideological direction of Supreme Court decisions: ensemble vs. unified case-based model.
* [Supreme Court Topic Modeling](https://github.com/AccelAI/AI-Law-Minicourse/tree/master/Supreme_Court_Topic_Modeling) - Multiple steps necessary to implement topic modeling on supreme court decisions. 
* [Judge Opinion](https://github.com/GirrajMaheshwari/Legal-Analytics-project---Court-misclassification) - Using text mining and machine learning to analyze judges’ opinions for a particular concern. 
* [ML Law Matching](https://github.com/whs2k/GPO-AI) - A machine learning law match maker.
* [Bert Multi-label Classification](https://github.com/brightmart/sentiment_analysis_fine_grain) - Fine Grained Sentiment Analysis from AI.
* [Some Computational AI Course](https://www.youtube.com/channel/UC5UHm2J9pbEZmWl97z_0hZw) - Video series Law MIT.

<a name="management"></a>
## Management

<a name="management-strat"></a>
#### Strategy
* [Topic Model Reviews](https://github.com/chrisjcc/DataInsight/blob/master/Topic_Analysis/Topic_modeling_Amazon_Reviews.ipynb) - Amazon reviews for product development. 
* [Patents](https://github.com/agdal1125/patent_analysis) - Forecasting strategy using patents.
* [Networks](https://github.com/JohnAnthonyBowllan/BusinessAI/blob/master/DataAnalysis_FeatureEngineering/businessCommunitiesMethod.ipynb) - Business categories from Yelp reviews using networks can help to identify pockets of demand. 
* [Company Clustering](https://github.com/DistrictDataLabs/company-clustering) - Hierarchical clusters and topics from companies by extracting information from their descriptions on their websites
* [Marketing Management](https://github.com/Jiseong-Michael-Yang/Marketing-Management) - Programmatic marketing management. 



<a name="management-do"></a>
#### Decision Optimisation
* [Constraint Learning](https://github.com/abrahami/Constraint-Learning) - Machine learning that takes into account constraints. 
* [Fairlearn](https://github.com/Microsoft/fairlearn) - I think it is called cost-sensitive machine learning.
* [Multi-label Classification](https://github.com/ej0cl6/csmlc) - Cost-Sensitive Multi-Label Classification
* [Multi-class Classification](https://github.com/david-cortes/costsensitive) - Cost-sensitive multi-class classification (Weighted-All-Pairs, Filter-Tree & others)
* [CostCla](http://albahnsen.github.io/CostSensitiveClassification/) - Costcla is a Python module for cost-sensitive machine learning (classification) built on top of Scikit-Learn
* [DEA Software](https://araith.github.io/pyDEA/) - pyDEA is a software package developed in Python for conducting data envelopment analysis (DEA).
* [Covering Set (FirmAI)](http://www.firmai.org/documents/Covering%20Set/) - Constraint programming analysis.
* [Insurance (FirmAI)](http://www.firmai.org/documents/Insurance/) - CP Insurance analysis.
* [Machine Learning + CP (FirmAI)](http://www.firmai.org/documents/MachineLearningand%20Optimisation/) - Machine Learning + Optimisation.
* [Post Office (FirmAI)](http://www.firmai.org/documents/Post%20Office/) - Post Office optimisation.
* [Soda - CP (FirmAI)](http://www.firmai.org/documents/soda_promotion-adapted-cp/) - Constraint Programming + ML.
* [Soda - Knapsack (FirmAI)](http://www.firmai.org/documents/soda_promotion-adapted-knapsack/) - Knapsack algorithm + ML.
* [Soda - MLP (FirmAI)](http://www.firmai.org/documents/soda_promotion-adapted-mip/) - MLP analysis + ML.

<a name="management-causal"></a>
#### Casual Inference
* [Marketing AB Testing](https://github.com/chrisjcc/DataInsight/tree/master/ABtesting) - A/B Testing Experiment.
* [Legal Studies](https://github.com/Akesari12/Intro_Causal_Inference) - Instrumental and discontinuity causal approach. 
* [A-B Test Result (FirmAI)](http://www.firmai.org/documents/Analyze_ab_test_results/) - Initial A-B Results.
* [Causal Regression (FirmAI) ](http://www.firmai.org/documents/causal_regression/) - Regression technique for causal estimate.
* [Frequentist vs Bayesian A-B Test (FirmAI)](http://www.firmai.org/documents/frequentist-bayesian-ab-testing/) - Comparison between frequentist and bayesian A-B testing.
* [A-B Test Power Analysis (FirmAI)](http://www.firmai.org/documents/Power%20analysis%20for%20AB%20tests/) - Sample size estimation to match testing power.
* [Variance Reduction A-B test (FirmAI)](http://www.firmai.org/documents/variance-reduction/) - Techniques to reduce variance in A-B tests.


<a name="management-stat"></a>
#### Statistics
* [Various](https://github.com/khanhnamle1994/applied-machine-learning/tree/master/Statistics) - Various applies statistical solutions 


<a name="management-quant"></a>
#### Quantitative 
* [Applied RL](https://github.com/mimoralea/applied-reinforcement-learning) - Reinforcement Learning and Decision Making tutorials explained at an intuitive level and with Jupyter Notebooks
* [Process Mining](https://github.com/yesanton/Process-Sequence-Prediction-with-A-priori-knowledge) - Leveraging A-priori Knowledge in Predictive Business Process Monitoring
* [TS Forecasting](https://github.com/khanhnamle1994/applied-machine-learning/tree/master/Time-Series-Forecasting) - Time series forecasting for important business applications.

####

<a name="management-data"></a>
#### Data 
* [Web Scraping (FirmAI)](www.firmai.org/data/) - Web scraping solutions for Facebook, Glassdoor, Instagram, Morningstar, Similarweb, Yelp, Spyfu, Linkedin, Angellist. 




<a name="operations"></a>
## Operations

<a name="operations-fail"></a>
#### Failure and Anomalies
* [Anomalies](https://github.com/yzhao062/anomaly-detection-resources) - Anomaly detection resources. 
* [Intrusion Detection](https://nbviewer.jupyter.org/github/albahnsen/ML_SecurityInformatics/blob/master/exercises/05-IntrusionDetection.ipynb) - Detecting network intrusions. 
* [APS Failure](https://github.com/Nisarg9795/Anomaly-Detection-APS-failures-in-Scania-trucks/blob/master/1_LR_Final_Code.py ), [Data](https://archive.ics.uci.edu/ml/datasets/APS+Failure+at+Scania+Trucks) - Investigating APS failures in Scania trucks. 
* [Hardware Failure](https://github.com/AbertayMachineLearningGroup/machine-learning-SIEM-water-infrastructure) - Using different machine learning techniques in detecting anomalies.
* [Anomaly KIs](https://github.com/haowen-xu/donut),[Paper](https://arxiv.org/abs/1802.03903)  - Anomaly detection algorithm for seasonal KPIs.

<a name="operations-load"></a>
#### Load and Capacity Management
* [House Load Energy](https://github.com/giorgosfatouros/Appliances-Energy-Load-Prediction) - Linear, SVR and Random Forest models to predict house's appliances energy Load.
* [Uber Load Management](https://github.com/brianallen131/Uber-Predictive-Load-Management) - Uber predictive load management.
* [Capacity Management](https://github.com/nerdiejack/capacity_management/blob/master/notebooks/MyWebshopAssignmentWithSolution.ipynb) - Investigating IT stability issues are caused by capacity constraints.
* [Bike Sharing](https://github.com/chrisjcc/DataInsight/blob/master/DataChallenge/BikeShare_Challenge.ipynb) - XGBRegressor, RandomForestRegressor, GradientBoostingRegressor combined with feature selection.
* [Airline Fleet Segmentation](http://htmlpreview.github.io/?https://github.com/atul-shukla-INSEAD/GroupProjectBDA/blob/master/GroupProject.html) - Analysis of Delta airlines.
* [Airbnb](http://inseaddataanalytics.github.io/INSEADAnalytics/groupprojects/AirbnbReport2016Jan.html) - Airbnb Booking Analysis.


<a name="operations-predict"></a>
#### Prediction Management
* [Dispute Prediction](https://github.com/zhanghaizhen/Financial-Service-Complaint-Management/tree/master/ipynb) - Financial service complaint management. 
* [Fight Delay Prediction](https://github.com/cavaunpeu/flight-delays/blob/master/notebooks/flight-prediction.ipynb) - Transfer learning for flight-delay prediction via variational autoencoders in Keras.
* [Electric Fault Prediction](https://github.com/susano0/Electric-Fault-Prediction/blob/master/Fault_pred.ipynb) - Predict tripping at grid stations by applying simple machine learning algorithms.
* [Popularity Prediction in R](https://github.com/s-mishra/featuredriven-hawkes/blob/master/code/marked_hawkes_point_process.ipynb) - Marked Hawkes Point Process .

<a name="public"></a>
## Public

<a name="public-social"></a>
#### Social Policies
* [Triage](https://github.com/dssg/triage ) - General Purpose Risk Modeling and Prediction Toolkit for Policy and Social Good Problems.
* [World Bank Poverty I](https://github.com/worldbank/ML-classification-algorithms-poverty/tree/master/notebooks) - A comparative assessment of machine learning classification algorithms applied to poverty prediction.
* [World Bank Poverty II](https://github.com/avsolatorio/world-bank-pover-t-tests-solution) - Repository for the World Bank Pover-t Test Competition Solution
Overseas Company Land Ownership .
* [Overseas Company Land Ownership](https://github.com/Global-Witness/overseas-companies-land-ownership/blob/master/overseas_companies_land_ownership_analysis.ipynb) - Identifying foreign ownership in the UK.
* [CFPB](https://github.com/MAydogdu/ConsumerFinancialProtectionBureau/blob/master/CFPB_Complaints_2017September.ipynb) - Consumer Finances Protection Bureau complaints analysis.
* [Cannabis Legalisation Effect](https://github.com/tslindner/Effects-of-Cannabis-Legalization) - Effects of cannabis legalization on crime. 

<a name="public-election"></a>
#### Election Analysis
* [Election Analysis](https://github.com/1jinwoo/DeepWave/blob/master/DR_Random_Forest.ipynb) - Election Analysis and Prediction Models
* [American Election Causal](https://github.com/Akesari12/LS123_Data_Prediction_Law_Spring-2019/blob/master/labs/OLS%20for%20Causal%20Inference/OLS_Causal_Inference_solution.ipynb) - Using ANES data with causal inference models.
* [Campaign Finance and Election Results](https://github.com/sfbrigade/datasci-campaign-finance/blob/master/notebooks/ML%20Campaign%20Finance%20and%20Election%20Results%20Example.ipynb) - Investigating the relation between campaign finance and subsequent election results. 


<a name="public-dis"></a>
#### Disaster Management
* [Conflict Prediction](https://github.com/Polichinel/Master_Thesis) - Notebooks on conflict prediction.
* [Burglary Prediction](https://github.com/Polichinel/Master_Thesis) - Spatio-Temporal Modelling for burglary prediction. 
* [Predicting Disease Outbreak](https://github.com/ab-bh/Disease-Outbreak-Prediction/blob/master/Disease%20Outbreak%20Prediction.ipynb) - Machine Learning implementation based on multiple classifier algorithm implementations.
* [Road accident prediction](https://github.com/leportella/federal-road-accidents) - Prediction on type of victims on federal road accidents in Brazil.
* [Text Mining](https://github.com/rajaswa/Disaster-Management-) - Disaster Management using Text mining.
* [Twitter and disasters](https://github.com/paultopia/concrete_NLP_tutorial/blob/master/NLP_notebook.ipynb) - Try to correctly predict whether tweets that are about disasters..

<a name="public-urban"></a>
#### Urban Planning
* [Traffic Prediction](https://github.com/mratsim/McKinsey-SmartCities-Traffic-Prediction) - Multi attention recurrent neural networks for time-series (city traffic) 
* [Predict Crashes](https://github.com/Data4Democracy/crash-model) - Crash prediction modeling application that leverages multiple data sources.
* [Predict Household Poverty](https://github.com/Featuretools/predict-household-poverty) - Predict the poverty of households in Costa Rica using automated feature engineering.


<a name="public-util"></a>
#### Utilities
* [Air Quality Prediction](https://github.com/txytju/air-quality-prediction) - Predict air quality(aq) in Beijing and London in the next 48 hours.
* [Water Accounting](https://github.com/johnpfay/USWaterAccounting) - Assembles water budget data for the US from existing data source.
* [Electricity French Distribution](http://inseaddataanalytics.github.io/INSEADAnalytics/groupprojects/group_energy.html) - An analysis of electricity data provided by the French Distribution Network (RTE).


