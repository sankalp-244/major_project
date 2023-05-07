# STEP -1 TAKE THE DATASET AND CREATE DATAFRAME
import  pandas as pd

df= pd.read_csv("https://raw.githubusercontent.com/ameenmanna8824/DATASETS/main/CarPrice_Assignment.csv")
print(df)

# STEP -2 EDA
df.info()
print(df.size)
print(df.shape)

#STEP - 3 VISUALISE THE DATA (NOT REQUIRED IN THIS CASE)

# STEP - 4 DIVIDE THE DATA INTO INOUT AND OUTPUT

# INPUT - wheelbase, carlength, carwidth, carheight, curbweight, boratio, stoke, compressionratio,horsepower,peakrpm,citympg,highwaympg
# compressratio, horsepower, peakrpm, citympy, highwaympg, price
# OUTPUT - CAR_NAME
# INPUT IS DENOTED BY 'x'
# OUTPUT IS DONETED BY 'y'

x = df[['wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight', 'enginesize','boreratio', 'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg']].values
y = df['price'].values
print(x)
print(y)


# STEP 5 TRAIN AND TEST VARIABLES

# FOR THIS WE WILL CONSIDER 4 VARIABLES INPUT_TRAINING, OUTPUT_TRAINING, INPUT_TESTING AND OUTPUT_TESTING

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 0)

# Size comparision of main varibles and their training and testing

print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_train.shape)
print(y_test.shape)

# STEP 6 NORMALISATION

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# STEP 7 - APPLY A CLASSIFIER, REGRESSOR OR CLUSTERER

from sklearn.linear_model import LinearRegression
model = LinearRegression()

# STEP 8 - TRAIN THE MODEL
model.fit(x_train, y_train)

# STEP 9 - PREDICT THE VALUES
y_pred = model.predict(x_test)
print(y_pred)
print(y_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)
print("Scoer is :", score*100)
