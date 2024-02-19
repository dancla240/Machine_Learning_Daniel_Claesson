import numpy as np
import pandas as pd
import scipy

def create_sample_data_simple_regression(size):
    '''Skapar sample data för enkel linjär regression, endast ett x värde.'''
    np.random.seed(42)
    x = np.abs(np.random.normal(loc=100, scale=100, size=size))
    error = np.random.normal(loc=0, scale=50, size=size)
    y = 2*x + 25 + error
    return x, y

def clean_sample_data_simple_regression(x, y):
    '''Cleans the data according to instructions in exercise\n
    All x-values >300 are deleted. All y values <0 are deleted.'''
    x_to_delete = np.where(x>300) # Skapar en array med alla index för x > 300.
    x = np.delete(x, x_to_delete) # tar bort alla värden ur x arrayen, med värden >300
    y = np.delete(y, x_to_delete) # tar bort alla värden ur y arrayen, där x > 300

    y_to_delete = np.where(y<0) # skapar en array med alla index för y < 0
    x = np.delete(x, y_to_delete) # tar bort alla x värden, där y < 0
    y = np.delete(y, y_to_delete) # tar bort alla y-värden, där y < 0

    return x, y

def train_test_split_simple_regression(data_frame, train_fraction = .7, random_state = 42, replace=False):
    '''Gör en split av datan till ett training dataset och ett test dataset. För simpel linjär regression.'''
    
    df_train = data_frame.sample(frac=train_fraction, random_state = random_state)
    df_test = data_frame.drop(index=list(df_train.index))

    X_train = df_train['Minutes']
    X_test = df_test['Minutes']
    y_train = df_train['Cost']
    y_test = df_test['Cost']

    return (X_train, y_train, X_test, y_test)

def calc_of_statistics_metrics(y_hat, y_test):
    '''Beräknar statistiska metrics över regression fit.'''
    n = len(y_test)
    MAE = 1 / n * np.sum(abs(y_hat - y_test)) # MAE: Mean Absolute Error
    MSE = 1 / n * np.sum((y_hat - y_test)**2) # MSE: Mean Squared Error
    RMSE = np.sqrt(MSE) # Root Mean Squared Error

    return MAE, MSE, RMSE

def create_sample_data_multiple_regression(sample_size):
    '''funktion för skapa sample data, för att kunna kalla på den för att göra den sista uppgiften'''
    np.random.seed(42)
    x1 = np.random.normal(loc=100, scale=100, size = sample_size,)
    x2 = np.random.uniform(low=0, high=50, size=sample_size)
    x3 = np.random.normal(loc=0, scale=2, size=sample_size)
    epsilon = np.random.normal(loc=0, scale=50, size=sample_size)
    y = 25 + 2*x1 + 0.5*x2 + 50*x3 + epsilon

    df_xs = pd.DataFrame(data = [x1, x2, x3, epsilon, y]).T
    df_xs.rename(columns={0:'Minutes',1:'SMS',2:'Data',3:"Epsilon", 4:"Cost"}, inplace=True)
    df_xs.insert(0,'Intercept', value=1)
    
    return df_xs #df_cleaned1 # returnerar en Pandas DataFrame med x-värden (inklusive intercept) och y värdet ('Cost')

def clean_sample_data_multiple_regression(data_frame):
    '''Rensar dataframen från orimlig data enligt instruktionen i uppgiften.'''
    df_cleaned = data_frame[(data_frame['Minutes'] <300) & (data_frame['Cost'] > 0) & (data_frame['Data'] < 4)]
    
    return df_cleaned # returnerar en städad Pandas DataFrame med x-värden (inklusive intercept) och y värdet ('Cost')

def train_test_split_multiple_regression(data_frame, train_fraction = .7, random_state = 42, replace=False):
    '''funktion för att dela in datan i training data och test data.'''

    np.random.seed(random_state)
    df_train = data_frame.sample(frac=train_fraction)
    df_test = data_frame.drop(index=list(df_train.index))

    X_train = np.array(df_train[['Intercept','Minutes','SMS','Data']]) # gör om till array så att man kan göra matrismultiplikation på den
    X_test = np.array(df_test[['Intercept','Minutes','SMS','Data']]) # gör om till array så att man kan göra matrismultiplikation på den

    y_train = np.array(df_train['Cost']) # gör om till array så att man kan göra matrismultiplikation på den
    y_test = np.array(df_test['Cost']) # gör om till array så att man kan göra matrismultiplikation på den

    return (X_train, y_train, X_test, y_test)

def regression_fit_scipy_multiple_regression(X_train, y_train):
    '''Calculation of best regression fit using the training data.\n
    dvs, beta_hat är vektorn med x1 x2 osv.'''
    beta_hat, residuals, rank, s = scipy.linalg.lstsq(X_train, y_train)

    return beta_hat

def calc_y_hat_multiple_regression(beta_hat, X_test):
    '''Beräknar predikterade y-värden (y_hat) baserat på regression\n
    fit (baserad på training data) och X_test-värden'''
    y_hat = np.array([])

    for idx in range(len(X_test)):
        to_add = beta_hat[0] + beta_hat[1] * X_test[idx][1] + beta_hat[2] * X_test[idx][2] + beta_hat[3] * X_test[idx][3]
        y_hat = np.append(y_hat, to_add)

    return y_hat