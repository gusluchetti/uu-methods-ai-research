import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


restaurants = pd.read_csv(r'C:\Users\niels\restaurant_info(1).csv')
type(restaurants['postcode'][20])
pd.isna(restaurants['postcode'][20])
print(restaurants)
#example_preferences = ['moderate', 'centre', 'british']
example_preferences = ['expensive', 'centre', 'korean']
remaining_restaurants = ''
index_reset_count = 0
# Search rows for pricerange, area, food.
# return the row that matches
# If multiple rows match, return at random, and store the other restaurants.
# Everything, only phone, only postcode, none
# restaurant,pricerange,area,food,phone number,addr, postcode

# RUN this function when the users wants to see additional restaurant recommendations
def additional_restaurants():
    #df.reset_index(level=0)
    drop_indices = np.random.choice(len(globals()['remaining_restaurants'].index))
    globals()['remaining_restaurants'] = globals()['remaining_restaurants'].reset_index(drop=True)
    globals()['remaining_restaurants'].drop(drop_indices)
    globals()['index_reset_count'] += 1
    if  (pd.isna(globals()['remaining_restaurants'].iloc[drop_indices][5])) and  (pd.isna(globals()['remaining_restaurants'].iloc[drop_indices][7])):

        return "We recommend the restaurant"+" "+str(globals()['remaining_restaurants'].iloc[drop_indices][1])+ '. '+'It is a '+str(globals()['remaining_restaurants'].iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(globals()['remaining_restaurants'].iloc[drop_indices][4]+ ' food. '+'It is located in the '+str(globals()['remaining_restaurants'].iloc[drop_indices][3])+' on '+str(globals()['remaining_restaurants'].iloc[drop_indices][6])+'.')

    elif(pd.isna(globals()['remaining_restaurants'].iloc[drop_indices][5])):

        return "We recommend the restaurant"+" "+str(globals()['remaining_restaurants'].iloc[drop_indices][1])+ '. '+'It is a '+str(globals()['remaining_restaurants'].iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(globals()['remaining_restaurants'].iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(globals()['remaining_restaurants'].iloc[drop_indices][3])+' on '+str(globals()['remaining_restaurants'].iloc[drop_indices][6])+' ('+str(globals()['remaining_restaurants'].iloc[drop_indices][7])+').'

    elif(pd.isna(globals()['remaining_restaurants'].iloc[drop_indices][7])):

        return "We recommend the restaurant"+" "+str(globals()['remaining_restaurants'].iloc[drop_indices][1])+ '. '+'It is a '+str(globals()['remaining_restaurants'].iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(globals()['remaining_restaurants'].iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(globals()['remaining_restaurants'].iloc[drop_indices][3])+' on '+str(globals()['remaining_restaurants'].iloc[drop_indices][6])+'. '+'The phonenumber of the restaurant is: '+str(globals()['remaining_restaurants'].iloc[drop_indices][5])+'.'

    else:
        return "We recommend the restaurant"+" "+str(globals()['remaining_restaurants'].iloc[drop_indices][1])+ '. '+'It is a '+str(globals()['remaining_restaurants'].iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(globals()['remaining_restaurants'].iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(globals()['remaining_restaurants'].iloc[drop_indices][3])+' on '+str(globals()['remaining_restaurants'].iloc[drop_indices][6])+' ('+str(globals()['remaining_restaurants'].iloc[drop_indices][7])+'). '+'The phonenumber of the restaurant is: '+str(globals()['remaining_restaurants'].iloc[drop_indices][5])+'.'





def find_best_restaurant(df, preferences, remaining_restaurants):
    globals()['index_reset_count'] = 0
    preferences = list(map(str.lower,preferences))
    if(len(preferences) == 3):
        result = df.loc[(df["pricerange"] == preferences[0]) & (df["area"] == preferences[1]) & (df["food"] == preferences[2])]
        result = result.reset_index()
    else:
        return 'Sorry, there is no restaurant that conforms to you requirements.'
    if(result.shape[0] > 1):

        # 5,7
        #drop_indices = np.random.choice(result.index, 1, replace=False)
        drop_indices = np.random.choice(len(result.index))
        print(pd.isna(result.iloc[drop_indices][7]))
        print(pd.isna(result.iloc[drop_indices][5]))
        print(result.iloc[drop_indices])
        globals()['remaining_restaurants'] = result.drop(drop_indices)
        if  (pd.isna(result.iloc[drop_indices][5])) and  (pd.isna(result.iloc[drop_indices][7])):
            
            return "We recommend the restaurant"+" "+str(result.iloc[drop_indices][1])+ '. '+'It is a '+str(result.iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(result.iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(result.iloc[drop_indices][3])+' on '+str(result.iloc[drop_indices][6])+'.'

        elif(pd.isna(result.iloc[drop_indices][5])):
            
            return "We recommend the restaurant"+" "+str(result.iloc[drop_indices][1])+ '. '+'It is a '+str(result.iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(result.iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(result.iloc[drop_indices][3])+' on '+str(result.iloc[drop_indices][6])+' ('+str(result.iloc[drop_indices][7])+').'
                
        elif(pd.isna(result.iloc[drop_indices][7])):
            
            return "We recommend the restaurant"+" "+str(result.iloc[drop_indices][1])+ '. '+'It is a '+str(result.iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(result.iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(result.iloc[drop_indices][3])+' on '+str(result.iloc[drop_indices][6])+'. '+'The phonenumber of the restaurant is: '+str(result.iloc[drop_indices][5])+'.'

        else:
            return "We recommend the restaurant"+" "+str(result.iloc[drop_indices][1])+ '. '+'It is a '+str(result.iloc[drop_indices][2])+' priced restaurant'+' that serves '+str(result.iloc[drop_indices][4])+ ' food. '+'It is located in the '+str(result.iloc[drop_indices][3])+' on '+str(result.iloc[drop_indices][6])+' ('+str(result.iloc[drop_indices][7])+'). '+'The phonenumber of the restaurant is: '+str(result.iloc[drop_indices][5])+'.'

        #df_subset = result.drop(drop_indices)
        #remove_row = np.random.seed(result.shape[0])
    

    if result.empty:
        return 'Sorry, there is no restaurant that conforms to you requirements.'
    elif result['phone'].isnull().values.any() and result['postcode'].isnull().values.any():
        
        return "We recommend the restaurant"+" "+str(result.iloc[0][1])+ '. '+'It is a '+str(result.iloc[0][2])+' priced restaurant'+' that serves '+str(result.iloc[0][4])+ ' food. '+'It is located in the '+str(result.iloc[0][3])+' on '+str(result.iloc[0][6])+'.'

    elif result['phone'].isnull().values.any():
        
        return "We recommend the restaurant"+" "+str(result.iloc[0][1])+ '. '+'It is a '+str(result.iloc[0][2])+' priced restaurant'+' that serves '+str(result.iloc[0][4])+ ' food. '+'It is located in the '+str(result.iloc[0][3])+' on '+str(result.iloc[0][6])+' ('+str(result.iloc[0][7])+').'

    elif result['postcode'].isnull().values.any():
        
        return "We recommend the restaurant"+" "+str(result.iloc[0][1])+ '. '+'It is a '+str(result.iloc[0][2])+' priced restaurant'+' that serves '+str(result.iloc[0][4])+ ' food. '+'It is located in the '+str(result.iloc[0][3])+' on '+str(result.iloc[0][6])+'. '+'The phonenumber of the restaurant is: '+str(result.iloc[0][5])+'.'

    else:
        return "We recommend the restaurant"+" "+str(result.iloc[0][1])+ '. '+'It is a '+str(result.iloc[0][2])+' priced restaurant'+' that serves '+str(result.iloc[0][4])+ ' food. '+'It is located in the '+str(result.iloc[0][3])+' on '+str(result.iloc[0][6])+' ('+str(result.iloc[0][7])+'). '+'The phonenumber of the restaurant is: '+str(result.iloc[0][5])+'.'

find_best_restaurant(restaurants, example_preferences,"")

additional_restaurants()

# If there is no restaurant that conforms to the requirements then the system should inform the user
# that there is no restaurant that conforms to the requirements of the user.
#
# If there are multiple possible restaurants then the system should select one of these at random and store
# the remaining matches in case the user asks for an alternative suggestion using the same preferences.
#
# Take into account that the input is converted to lower case when implementing the lookup function.

# - no restaurant case
# - edge cases
# - multiple restaurants case

# We recommend the restaurant restaurantname, it is an pricerange restaurant that serves food.
# It is located in the area area on addr postcode. The phonenumber of the restaurant is: phone.
