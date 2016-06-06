#####################################################
# Tyler Hedegard
# 6/6/2016
# Thinkful Data Science
# Lending Data (Logistic Regression)
#####################################################

import pandas as pd
import statsmodels.api as sm
import math

loansData = pd.read_csv('loansData_clean.csv')
df = loansData

# Statement in the lesson:
# Add a column to your dataframe indicating whether the interest rate is < 12%.
# This would be a derived column that you create from the interest rate column.
# You name it IR_TF.
# It would contain binary values, i.e.'0' when interest rate < 12%
# or '1' when interest rate is >= 12%

# found a nifty if/else syntax for lambda: http://eikke.com/python-ifelse-in-lambda/
# lambda x: <condition> and <true value> or <false value>
# Had to switch the true/false values to make the sanity checks pass
df['IR_TF'] = df['Interest.Rate'].map(lambda x: x < 12 and 1 or 0)
df['intercept'] = 1.0

# These contradict the statement in the lesson:
# df[df['Interest.Rate'] == 10].head() # should all be True
# ^^ This is when x < 12, but the lesson says to make that '0'
# df[df['Interest.Rate'] == 13].head() # should all be False
# ^^ This is when x >=12, but the lesson says to make that '1'

ind_vars = ['Amount.Requested', 'FICO.Score','intercept']
logit = sm.Logit(df['IR_TF'], df[ind_vars])
result = logit.fit()
coeff = result.params


def logistic_function(score, amount, coeff):
    interest_rate = coeff['intercept'] + coeff['FICO.Score']*score + coeff['Amount.Requested']*amount
    p = 1/(1 + math.e**interest_rate)
    return p

logistic_function(720, 10000, coeff)
# p = 0.254
# p < 0.7 therefore we will not get the loan

df.plot(kind='scatter', x='FICO.Score', y='Amount.Requested');