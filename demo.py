#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


# In[2]:


def get_nta_input(df) -> str:
    try:
        nta_options = df['NTA'].unique()[:10]
        
        nta_input = str(input(fr'''
Enter your current NTA.
Some potential options include {nta_options}:

        '''))
        while nta_input not in df['NTA'].unique().astype(str):
            nta_input = str(input(fr'''
That was an invalid input.

Enter your current NTA.
Some potential options include {nta_options}:

            '''))
    except Exception as e:
        print('Your input was not of the correct type.')
        return get_nta_input()
    
    return nta_input


# In[3]:


def get_is_black_input() -> str:
    try:
        is_black_input = str(input('''
Are you black? (Y/N)

    ''')).upper()
        while is_black_input not in ['Y', 'N']:
            is_black_input = str(input(fr'''
That was an invalid input.

Are you black? (Y/N)

            ''')).upper()
        
    except Exception as e:
        print('Your input was not of the correct type.')
        return get_is_black_input()
    
    return is_black_input


# In[4]:


def get_gender_input() -> str:
    try:
        gender_input = str(input('''
Are you male or female? (M for male/F for female)

    ''')).upper()
        while gender_input not in ['M', 'F']:
            gender_input = str(input(fr'''
That was an invalid input.

Are you male or female? (M for male/F for female)

            ''')).upper()
        
    except Exception as e:
        print('Your input was not of the correct type.')
        return get_gender_input()
    
    return gender_input


# In[5]:


def get_age_input() -> int:
    try:
        age_input = int(input('''
What age are you?

    '''))
        while age_input not in np.arange(0, 120):
            age_input = str(input(fr'''
That was an invalid input.

What age are you?

            '''))
            
    except Exception as e:
        print('Your input was not of the correct type.')
        return get_age_input()
    
    return age_input


# In[6]:


def get_time_of_day_input() -> int:
    try:
        time_of_day_input = int(input('''
What time of day is it? (Type a single number representing the current hour in military time [0-23].)

    '''))
        while time_of_day_input not in np.arange(0, 24):
            time_of_day_input = int(input(fr'''
That was an invalid input.

What time of day is it? (Type a single number representing the current hour in military time [0-23].)

            '''))
            
    except Exception as e:
        print('Your input was not of the correct type.')
        return get_time_of_day_input()
    
    return time_of_day_input


# In[7]:


def get_month_input() -> int:
    try:
        month_input = int(input('''
What month is it? (Give a number between 1 and 12 representing the month [1-12].)

    '''))
        while month_input not in np.arange(1, 13):
            month_input = int(input(fr'''
That was an invalid input.

What month is it? (Give a number between 1 and 12 representing the month [1-12].)

            '''))
    
    except Exception as e:
        print('Your input was not of the correct type.')
        get_month_input()
    
    return month_input


# In[8]:


def get_continue_or_not_input() -> str:
    try:
        continue_or_not_input = str(input('''
Would you like to continue or not? (Y/N)

    ''')).upper()
        while continue_or_not_input not in ['Y', 'N']:
            continue_or_not_input = int(input(fr'''
That was an invalid input.

Would you like to continue or not? (Y/N)

            ''')).upper()
    
    except Exception as e:
        print('Your input was not of the correct type.')
        get_continue_or_not_input()
    
    return continue_or_not_input


# In[9]:


def get_user_data(df) -> pd.DataFrame():
    nta_input = get_nta_input(df)
    is_black_input = get_is_black_input()
    gender_input = get_gender_input()
    age_input = get_age_input()
    time_of_day_input = get_time_of_day_input()
    month_input = get_month_input()
    
    user_profile = pd.DataFrame([[nta_input, is_black_input, gender_input, age_input, time_of_day_input, month_input]], columns = df.columns[:6])
    
    return user_profile


# In[10]:


def preprocess_is_black(df):
    df['is_black'] = np.where(df['is_black'] == 'Y', 1, 0)
    return df


# In[11]:


def preprocess_victim_sex(df):
    df['victim_sex'] = np.where(df['victim_sex'] == 'M', 0, 1)
    return df


# In[12]:


def preprocess_victim_age(df):
    df.loc[df['victim_age_range'] < 18, 'victim_age_range'] = 0
    df.loc[((df['victim_age_range'] >= 18) & (df['victim_age_range'] <= 24)), 'victim_age_range'] = 1
    df.loc[((df['victim_age_range'] >= 25) & (df['victim_age_range'] <= 44)), 'victim_age_range'] = 2
    df.loc[((df['victim_age_range'] >= 45) & (df['victim_age_range'] <= 64)), 'victim_age_range'] = 3
    df.loc[df['victim_age_range'] >= 65, 'victim_age_range'] = 4
    
    return df


# In[13]:


def preprocess_hour(df):
    df.loc[df['hour'].isin(np.arange(0, 6)), 'hour'] = 0
    df.loc[df['hour'].isin(np.arange(6, 12)), 'hour'] = 1
    df.loc[df['hour'].isin(np.arange(12, 18)), 'hour'] = 2
    df.loc[df['hour'].isin(np.arange(18, 25)), 'hour'] = 3
    
    return df


# In[14]:


def preprocess_month(df):
    df.loc[df['month'].isin(np.arange(1, 4)), 'month'] = 0
    df.loc[df['month'].isin(np.arange(4, 7)), 'month'] = 1
    df.loc[df['month'].isin(np.arange(7, 10)), 'month'] = 2
    df.loc[df['month'].isin(np.arange(10, 13)), 'month'] = 3
    
    return df


# In[31]:


def plot_data(df, df_2):
    demographic = df_2.values.tolist()[0]
    
    features = np.arange(2014, 2022).astype(str).tolist() + ['forecast']
    hours = [fr'{i % 12}:00 AM' if i < 12 else fr'{i % 12}:00 PM' for i in range(0, 24)]
    months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    
    sns.set(rc = {'figure.figsize': (20, 10)})
    
    ax = sns.barplot(data = df[features])
    ax.set_title(fr'''
Historical Danger Score and Forecast for Demorgraphic:
NTA: {demographic[0]}, Black or Not?: {demographic[1]}, Gender: {demographic[2]}, Age: {str(demographic[3])}, Hour: {hours[demographic[4]]}, Month: {months[demographic[5]]}
    '''
    , fontsize = 24
    )
    ax.set_xlabel('Year', fontsize = 24)
    ax.set_ylabel('Danger Score', fontsize = 24)
    plt.show()


# In[32]:


def main():
    print('''
Welcome to the danger score forecaster!

Based on your answers to these questions, we will predict how dangerous a walk in New York will be for you.

Here are some answers that will give you interesting results:
{'NTA': BK35, 'is_black': Y, 'gender': F, 'age': 30, 'hour': 14, 'month': 8}
{'NTA': BK72, 'is_black': N, 'gender': M, 'age': 35, 'hour': 21, 'month': 5}
{'NTA': QN71, 'is_black': N, 'gender': F, 'age': 39, 'hour': 23, 'month': 9}

    ''')
    forecast_data = 'crime_prediction_forecast_table'
    file_directory = os.path.abspath(forecast_data)
    df_forecast = pd.read_csv(file_directory, index_col = [0])
    
    keep_going = True
    while keep_going == True:
        keep_going = False
        
        try:
            user_profile_original = get_user_data(df_forecast)

            user_profile = preprocess_is_black(user_profile_original.copy())
            user_profile = preprocess_victim_sex(user_profile)
            user_profile = preprocess_victim_age(user_profile)
            user_profile = preprocess_hour(user_profile)
            user_profile = preprocess_month(user_profile)

            df_forecast[['is_black', 'victim_sex', 'victim_age_range', 'hour', 'month']] = df_forecast[['is_black', 'victim_sex', 'victim_age_range', 'hour', 'month']].astype(int)
            user_profile[['is_black', 'victim_sex', 'victim_age_range', 'hour', 'month']] = user_profile[['is_black', 'victim_sex', 'victim_age_range', 'hour', 'month']].astype(int)

            user_data = pd.merge(df_forecast, user_profile,  how = 'inner', on = ['NTA', 'is_black', 'victim_sex', 'victim_age_range', 'hour', 'month'])

            plot_data(user_data, user_profile_original)
        except Exception as e:
            print('Something went wrong.')
        finally:
            continue_or_not = get_continue_or_not_input()

            if continue_or_not == 'Y':
                keep_going = True


# In[30]:


if __name__ == "__main__":
    main()


# In[ ]:




