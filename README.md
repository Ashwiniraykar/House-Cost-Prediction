# House-Cost-Prediction
To predict the price of a house based on a set of provided features.
## Dataset:
- Downloaded the dataset for this project from the Kaggle Website.
- The dataset consists of real property information, including most recent sales price as of July 2018, for properties located in Washington, D.C. 
- The total number of rows in the file is approximately 160,000. There are forty-nine columns (features) for each data sample.
- To make the data size more manageable, I focus only on the data with SOURCE=’Residential’. This will reduce the data size to about 107,000 rows.
## Implementation:
- Implemented a sensible approach for dealing with the missing data to get a high level of confidence in a predictive model. 
- Used linear regression to create a predictive model.
- Converted  the PRICE column into a categorical value, that means replaced the original value with one of ten (10) categories. For example, 0 will be substituted for the price      range of 0.0-30,000.  
- Used Logistic regression to create a predictive model using the modified data described in (c).
- Used a neural network to build a predictive model for the same data. 
