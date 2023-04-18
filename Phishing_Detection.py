# Logistic Regression for Fun: Phishing Detection

import zipfile
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import string
import collections as ct

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import words
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from collections import Counter

# initial loading, go to line 323 for modeling start.

df = pd.read_csv('phishing_site_urls.csv')

df.head()

len(df) # 548928 rows


# High Level Exploration and Cleaning of the Data Frame

"""
- here's a url that has some data cleaning code
    - https://www.kaggle.com/code/ashishkumarbehera/phishing-site-prediction
"""

len(df)


""" 
- (below) let's create a column where `1` is a URL identified as phishing, and `0` is a "good" url.
"""

df['phishing'] = (df.Label == 'bad').astype(int)

df[::10000]

df.drop('Label', axis = 1)

"""
- (below) let's look at the "good" urls and see if we need to do any data cleaning
"""

df[df['phishing'] == 0].iloc[::10000]

"""
- (below) we suspect there is a data quality issue.  Here is a URL that looks ok.
"""

df["URL"].iloc[0]

df["URL"].str.len().iloc[0]

normal_chars = string.printable

sum(v for k, v in ct.Counter(df["URL"].iloc[0]).items() if k in normal_chars)

"""
- (below) here is a URL that we are concerned about.
"""
df["URL"].iloc[18232]

df["URL"].str.len().iloc[18234]

"""
- (below) a function for identifying which rows contain these bad URLs.
"""

def non_printable_indices(col):
    # Returns a vector of indices for elements that contain non-printable characters.
    indices = []
    for i, s in col.items():
        if any(c not in string.printable for c in s):
            indices.append(i)
    return indices

bad_URL_rows = non_printable_indices(df["URL"])

bad_df = df.iloc[bad_URL_rows]

bad_df.head()

"""
- (below) the weird urls are a little less represented in the "non-phishing" category, but they represent a small portion of total rows so we're going to remove them.
"""

(bad_df['phishing'] == 1).sum()/(bad_df['phishing'] == 0).sum()

(df['phishing'] == 1).sum()/(df['phishing'] == 0).sum()

"""
- (below) the actual removal...
"""

df = df.drop(bad_URL_rows)

sns.countplot(x="phishing",data=df)

# Feature Engineering
"""
- (below) this is our first feature! a length count of characters.
"""

df['url_length'] = df['URL'].str.len()

df.head()

"""
- (below) create a tokenizer object that can be used to split text into tokens based on a regular expression pattern.
"""

tokenizer = RegexpTokenizer(r'[A-Za-z]+')

tokenizer.tokenize(df.URL[0]) 

"""
- (below) lets apply the tokenizer to every record in df.URL 
    - NOTE:  a lambda function is a small anonymous function that can be defined inline without a name. It's a shorthand way to define a function that takes arguments, performs an operation on them, and returns a value, all in a single line of code.
    - basic syntac is `add = lambda x, y: x + y`
    - `add(3, 5)` will return 8
"""

df['text_tokenized'] = df.URL.map(lambda t: tokenizer.tokenize(t))

"""
- (below) along with the tokenized column (which returns a list for each element, let's also create a column that holds concatenated string of the tokenized words instead.
"""

df['text_token_conc'] = df['text_tokenized'].map(lambda l: ' '.join(l))

# Jonathan's Features
"""
- (below) my first feature will be a count of the special characters in each URL.
"""

punct_chars = string.punctuation

df['punct_count'] = df['URL'].apply(lambda string: sum(1 for c in string if c in punct_chars))

"""
- (below) my second feature will count the words within each element of `df['text_tokenized']` that are also found in the english dictionary.
   - NOTE: the nltk library natively contains the words dictionary but may not be comprehensive or appropriate for all use cases. 
"""

nltk.download('wordnet')
nltk.download('omw-1.4')

df['num_english_words'] = df['text_tokenized'].apply(lambda tokens: sum([token in wordnet.words() for token in tokens]))

"""
- (above)  the .apply() method to apply a function that takes each list of tokens as input, and returns a scalar value representing the number of English words in the list.
- (above) we use .apply() instead of .map() because the input data is a DataFrame column that contains lists, not a Series of individual elements. 
   - The .apply() method can handle this type of input data, while the .map() method is designed for Series of individual elements.

- (below) my third feature will count the number of case changes in an URL.  I will normalize this by the length of the URL, which we already created earlier.
  - I will first remove all punctuation from the URL.
  - I will consider a case change to be a change from...
      - lower case to upper case.
      - lower case to number.
      - upper case to lower case.
      - upper case to number.
      - number to lower case.
      - number to upper case.
"""

def count_case_changes(string):
    no_punct = ''.join(c for c in string if c not in punct_chars and c != ' ')
    
    chg_counter = 0
    
    prev_char = ''
    
    for char in no_punct:
        if prev_char.islower() and char.isupper():
            chg_counter += 1
        elif prev_char.islower() and char.isdigit():
            chg_counter += 1
        elif prev_char.isupper() and char.islower():
            chg_counter += 1
        elif prev_char.isupper() and char.isdigit():
            chg_counter += 1
        elif prev_char.isdigit() and char.islower():
            chg_counter += 1
        elif prev_char.isdigit() and char.isupper():
            chg_counter += 1
        
        prev_char = char
    
    return chg_counter

df['case_change_count'] = df.URL.map(lambda string: count_case_changes(string))

"""
# - (below) save my data frame for later use so we don't need to run the english word check again!
"""
# In[48]:


df.head()


# In[12]:


df.to_pickle('phishing_df.pkl')


# ## Bryan Features 

# In[15]:


zip_filename = "phishing_df.zip"
pkl_filename = "phishing_df.pkl"

with zipfile.ZipFile(zip_filename) as z:
    with z.open(pkl_filename) as f:
        # load the PKL file into a pandas DataFrame
        df = pd.read_pickle(f)


# In[4]:

"""
- (below) Feature that determines whether www. is found in the string 
- (below) add a column that is a 1 if the url contains www and 0 otherwise 
"""

contains_www = df.assign(contains_www = df['URL'].apply(lambda x: 1 if 'www.' in x else 0))


# In[5]:

"""
- (below) Do the same but checks to see if the url starts with www. 
"""
startswith_www = df.assign(startswith_www = df['URL'].apply(lambda x: 1 if x.startswith('www.') else 0))


# In[6]:

"""
- (below) Returns the number of characters after www
- (below) filter down to strings that contain www. 
"""
contains_www_filtered = contains_www.query('contains_www == 1')

"""
- (below) add column with count of characters after www. 
"""
contains_www_filtered = contains_www_filtered.assign(count = contains_www_filtered['URL'].apply(lambda x: len(x.split('www.', 1)[-1])))

"""
- (below) median str len after www
"""

str_len_median = contains_www_filtered['count'].median()
str_len_median

"""
- (below) 1 if str length is above median and zero if below
"""

contains_www_filtered = contains_www_filtered.assign(above_median = contains_www_filtered['count'] > str_len_median)
contains_www_filtered['above_median'] = contains_www_filtered['above_median'].astype(int)

"""
- (below) Returns 1 if www is found more than once
"""

multiple_www = df.assign(mult_www=df['URL'].apply(lambda x: x.count('www.') > 1))
multiple_www['mult_www'] = multiple_www['mult_www'].astype(int)
sum(multiple_www['mult_www'] == 1)/len(multiple_www)

"""
- (below) assign new columns back to data frame.
"""
contains_www.head()

df['contains_www'] = contains_www['contains_www']

startswith_www.head()

df['startswith_www'] = startswith_www['startswith_www']

contains_www_filtered.head()

df['chars_past_www'] = contains_www_filtered['count']
df['chars_past_www_above_median'] = contains_www_filtered['above_median']

multiple_www.head()

df['multi_www'] = multiple_www['mult_www']

df.head()

df.fillna(0, inplace=True)
df['chars_past_www'] = df['chars_past_www'].astype(int)
df['chars_past_www_above_median'] = df['chars_past_www_above_median'].astype(int)

df.head()

"""
- (below) reorder columns
"""

df = df[['phishing', 'url_length', 'num_english_words','punct_count', 'case_change_count','contains_www', 'startswith_www', 'chars_past_www', 'chars_past_www_above_median', 'multi_www', 'URL', 'text_tokenized', 'text_token_conc', 'Label']]

df.head()

"""
- (below) resave to PKL file
"""
df.to_pickle('phishing_df.pkl')


# Modeling

"""
- (below) Loading the pickle file.
"""

zip_filename = "phishing_df.zip"
pkl_filename = "phishing_df.pkl"

with zipfile.ZipFile(zip_filename) as z:
    with z.open(pkl_filename) as f:
        # load the PKL file into a pandas DataFrame
        df = pd.read_pickle(f)

df.head()

"""
- (below) one last check for missing or NA values.
"""
null_counts = df.isnull().sum()

null_counts

na_counts = df.isna().sum()

na_counts

"""
- (below) let's remove the text/tokenized columns to prepare for modeling.
"""

df.columns

df_clean = df[['phishing', 'url_length', 'num_english_words','punct_count', 'case_change_count','contains_www', 'startswith_www', 'chars_past_www', 'chars_past_www_above_median', 'multi_www']]
df_clean.head()

df_clean.dtypes

# Train \ Val \ Test Split

"""
- (below) train \ val \ test split.
- (below) remember, don't look at test until the very end!
"""
df_full_train, df_test = train_test_split(df_clean, test_size=0.20, random_state=2023)

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=2023)

len(df_train)/len(df), len(df_val)/len(df), len(df_test)/len(df)

"""
- (below) optional, but reset the indices.
"""
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

"""
- (below) create response vectors. We use the .values method to get a numpy array which most modeling functions require.
"""

type(df_train.phishing), type(df_train.phishing.values)

y_train = df_train.phishing.values
y_val = df_val.phishing.values
y_test = df_test.phishing.values

"""
- (below) let's delete the response vector from these datasets so we don't accidently feed it into a model
- (below) we'll leave the phising variable in df_full_train in case we do some EDA.
"""

del df_train['phishing']
del df_val['phishing']
del df_test['phishing']

# Logistic Regression

"""
# - (below) initial training of logistic regression model
"""

df_train.head()

X_train = df_train.values

model = LogisticRegression(max_iter= 1000)
model.fit(X_train, y_train)

"""
- (below) bias term.
"""
w0 = model.intercept_[0] # its a 2d array, we only need 1 row tho.
w0

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

    bias = sigmoid(w0)

    print("bias: {}" .format(bias))


"""
- (above) without knowing anything about a URL, the probability it is a phishing site is thought to be 28.3%.
- (below) coefficients.
"""

model.coef_[0].round(3) # same

coefs = dict(zip(df_train.columns, model.coef_[0].round(3)))
coefs

# Predicting the validation set.

"""
- (below) hard predictions (0, 1)
"""

X_val = df_val.values
model.predict(X_val)

"""
- (below) soft predictions (probabilities)
- (below) column 1 is the probability of no phishing, column 2 is the probability of phishing.
"""

model.predict_proba(X_val)

y_val_pred = model.predict_proba(X_val)[:, 1]
y_val_pred

"""
- (below) our first decision rule will be a probability of 0.5
"""

phishing_decision = (y_val_pred >= 0.5)
phishing_decision

"""
- (below) our first accuracy measure
"""

(y_val == phishing_decision).mean()

# tune the decision threshold

thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_val_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)

plt.plot(thresholds, scores)

plt.xlabel("probability threshold for decision")

plt.ylabel("percent accuracy")

"""
- (above) looks like 0.45 is a better decision rule.
"""

phishing_decision = (y_val_pred >= 0.45)
phishing_decision

"""
- (below) let's make a prediction table
"""

df_pred = pd.DataFrame()
df_pred['probability'] = y_val_pred
df_pred['prediction'] = phishing_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred

df_pred.correct.mean()

accuracy_score(y_val, y_val_pred >= 0.45)

"""
- (below) just double-checking that the model was able to predict some phishing sites...
"""

df_pred[(df_pred['correct'] == True) & (df_pred['prediction'] == 1)]

"""
- (below) there shouldn't be in any predictions with a 1.0 probability.
"""

len(y_val_pred)

Counter(y_val_pred >= 1.0)

"""
- (below) we need to be aware of class balance.
"""

print('non-phishing sites: {}' .format(np.bincount(y_val)[0]))
print('phishing sites: {}' .format(np.bincount(y_val)[1]))
print('there are {:.2f} times as many non-phishing sites as phishing sites'.format(np.bincount(y_val)[0] / np.bincount(y_val)[1]))

"""
- (below) accuracy predicting phishing sites...
"""

df_pred[df_pred['actual'] == 1].correct.mean()

"""
- (below) accuracy predicting good sites...
"""
df_pred[df_pred['actual'] == 0].correct.mean()





