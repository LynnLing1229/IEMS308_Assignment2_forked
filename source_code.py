#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
pd.set_option('display.float_format', lambda x: '%.2f' % x)
import csv
import copy 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

import random


# ------

# # <u> 1. Data Preparing & Preprocessing</u>
# ## <i> 1.1. Labeling Columns </i>
# ### 1.1.1. TRNSACT
# All the lables of the columns are missing. Therefore, before any other steps, I started with giving labels to the data tables by taking a look at data schema and column description. I do not recommend running this part, because it will take a long time. 

# In[ ]:


trnsact = pd.read_csv("data/Dillards POS/trnsact.csv", header = None, usecols = range(0, 13),
                  names = ['SKU','STORE', 'REGISTER', 'TRANNUM', 'SEQ', 'SALEDATE','STYPE',
                           'QUANTITY', 'ORGPRICE', 'SPRICE', 'AMT', 'INTERID', 'MIC'])

# Creating trnsact_identifier which would be used to identify unique transactions
print("Step 1")
trnsact["trnsact_identifier"] = trnsact["STORE"].astype(str) + trnsact["REGISTER"].astype(str)
print("Step 2")
trnsact["trnsact_identifier"] = trnsact["trnsact_identifier"].astype(str) + trnsact["TRANNUM"].astype(str)
print("Step 3")
trnsact["trnsact_identifier"] = trnsact["SALEDATE"].astype(str) + trnsact["trnsact_identifier"].astype(str)

trnsact.head()


# In[ ]:


trnsact.to_csv("data/Dillards POS/trnsact_labelled.csv", index = False)


# ### 1.1.2. STRINFO

# In[ ]:


strinfo = pd.read_csv("data/Dillards POS/strinfo.csv", header = None, usecols=range(0, 4),
                  names = ["STORE", "CITY", "STATE", "ZIP"])
strinfo.head()


# In[ ]:


strinfo.to_csv("data/Dillards POS/strinfo_labelled.csv", index = False)


# ### 1.1.3. SKSTINFO

# In[ ]:


skstinfo = pd.read_csv("data/Dillards POS/skstinfo.csv", header = None, usecols=range(0, 4),
                  names = ["SKU", "STORE", "COST", "RETAIL"])
skstinfo.head()


# In[ ]:


skstinfo.to_csv("data/Dillards POS/skstinfo_labelled.csv", index = False)


# ### 1.1.4. DEPTINFO

# In[ ]:


deptinfo = pd.read_csv("data/Dillards POS/deptinfo.csv", header = None, usecols=range(0, 2),
                  names = ["DEPT", "DEPTDESC"])
deptinfo.head()


# In[ ]:


deptinfo.to_csv("data/Dillards POS/deptinfo_labelled.csv", index = False)


# ### 1.1.5. SKUINFO

# In[ ]:


skuinfo = pd.read_csv("data/Dillards POS/skuinfo.csv", header = None, usecols=range(0, 10),
                  names = ["SKU", "DEPT", "CLASSID", "UPC", "STYLE", "COLOR", "SIZE", "PACKSIZE", "VENDOR", "BRAND"])
skuinfo.head()


# In[ ]:


skuinfo.to_csv("data/Dillards POS/skuinfo_labelled.csv", index = False)


# ## <i> 1.2. Understanding the Structure of Data </i>
# To better understand the data I received, I scrutinized the structure of the data (checking the shape, missing values, distribution of values in each column, and unique values of each column)

# In[2]:


def structure_checker(data): 
    print("What is the shape of the data?")
    print(data.shape)
    print("========")
    print("Is there any missing data?")
    print(data.isnull().sum())
    print("========")    
    print("What is the distribution of data?")
    print(data.describe())
    print("========")
    print("What is number of unique values of each column of data?")
    for col in data.columns:
        print("Number of unique values of " + col + " column: " + str(data[col].nunique()))


# In[3]:


trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled.csv")
structure_checker(trnsact)


# In[4]:


strinfo = pd.read_csv("data/Dillards POS/strinfo_labelled.csv")
structure_checker(strinfo)


# In[5]:


skstinfo = pd.read_csv("data/Dillards POS/skstinfo_labelled.csv")
structure_checker(skstinfo)


# In[6]:


deptinfo = pd.read_csv("data/Dillards POS/deptinfo_labelled.csv")
structure_checker(deptinfo)


# In[7]:


skuinfo = pd.read_csv("data/Dillards POS/skuinfo_labelled.csv")
structure_checker(skuinfo)


# ## <i> 1.3. Filtering Data </i>
# To handle the realistic computation issues and to mitigate the bias caused by taking all STOREs with different sizes all together, filtering has been done to choose specific SKUs and STOREs for the analysis. Check the report for the rationale for each filtering step.

# In[ ]:


trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled.csv")
trnsact.shape


# ### 1.3.1. STYPE Selection

# In[8]:


Pcount = pd.DataFrame(trnsact[trnsact["STYPE"] == "P"].groupby(["SKU"])["QUANTITY"].agg('sum')).reset_index()
Pcount.columns = ["SKU", "P Count"]
Pcount.head()


# In[9]:


Rcount = pd.DataFrame(trnsact[trnsact["STYPE"] == "R"].groupby(["SKU"])["QUANTITY"].agg('sum')).reset_index()
Rcount.columns = ["SKU", "R Count"]
Rcount.head()


# In[10]:


merged = Pcount.merge(Rcount, how = "left")
merged.fillna(0, inplace = True)
merged["perc"] = 100.0*merged["R Count"]/merged["P Count"]
merged


# In[11]:


print("Percentage of SKUs with no return transaction: %3.2f%%" % (100.0*merged[merged["R Count"] == 0.00].shape[0]/merged.shape[0]))
print("Max(Return Quantity/Purchase Quantity Percentage): %3.2f%%" % max(merged[merged["R Count"] != 0.00]["perc"]))


# In[12]:


merged[merged["R Count"] != 0.00]["perc"].hist(bins =range(0, int(max(merged[merged["R Count"] != 0.00]["perc"]))))
plt.title("Distribution of return rate ")
plt.xlabel("return rate (%)")
plt.ylabel("number of SKUs")
plt.savefig('img/return rate.png')
plt.show()


# In[13]:


merged[merged["R Count"] != 0.00]["perc"].hist(bins = range(0, 100))
plt.title("Distribution of return rate (only 0% to 100%)")
plt.xlabel("return rate (%)")
plt.ylabel("number of SKUs")
plt.savefig('img/return rate  (only 0% to 100%).png')
plt.show()


# In[14]:


low_return_SKU = merged[merged["perc"] < 20.0]
low_return_SKU.shape


# In[15]:


filtered_trnsact = trnsact[trnsact["SKU"].isin(list(low_return_SKU["SKU"].unique()))]
filtered_trnsact = filtered_trnsact[filtered_trnsact["STYPE"] == "P"]
filtered_trnsact.shape


# In[16]:


filtered_trnsact.to_csv("data/Dillards POS/trnsact_labelled_filtered0.csv", index = False)


# ### 1.3.2. SALEDATE Selection

# In[ ]:


filtered_trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled_filtered0.csv")


# In[17]:


number_of_items_per_transaction = pd.DataFrame(filtered_trnsact.groupby(["trnsact_identifier"]).size().reset_index())
number_of_items_per_transaction.columns = ["trnsact_identifier", "number of items"]
number_of_items_per_transaction.head()


# In[18]:


one_item_per_transaction = number_of_items_per_transaction[number_of_items_per_transaction["number of items"] <= 1]
one_item_per_transaction.shape


# In[19]:


filtered_trnsact = filtered_trnsact[~filtered_trnsact["trnsact_identifier"].isin(one_item_per_transaction["trnsact_identifier"])]


# In[20]:


perday = pd.DataFrame(filtered_trnsact.groupby("SALEDATE")["trnsact_identifier"].nunique())
perday.head()


# In[21]:


perday.sort_values("trnsact_identifier", ascending = False)


# In[22]:


plt.rcParams['figure.figsize'] = [20, 5]
plt.plot(perday.index, perday["trnsact_identifier"])
plt.title("Number of transactions per day")
plt.xlabel("Date")
plt.ylabel("number of unique transactions")
plt.savefig('img/Number of transactions per day.png')
plt.show()


# In[23]:


filtered_trnsact = filtered_trnsact[filtered_trnsact["SALEDATE"] <= '2004-11-27']
filtered_trnsact = filtered_trnsact[filtered_trnsact["SALEDATE"] >= '2004-08-28']
filtered_trnsact.to_csv("data/Dillards POS/trnsact_labelled_filtered1.csv", index = False)


# ### 1.3.3. STORE Selection

# In[ ]:


filtered_trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled_filtered1.csv")


# In[24]:


filtered_trnsact["SKU"].nunique() # computationally heavy so cannot do all SKU


# In[25]:


common_sku = filtered_trnsact[filtered_trnsact["STORE"] == min(filtered_trnsact["STORE"].unique())]["SKU"].unique()
for i in filtered_trnsact["STORE"].unique():
    cur = filtered_trnsact[filtered_trnsact["STORE"]==i]["SKU"].unique()
    common_sku = set(common_sku) & set(cur)


# In[26]:


len(common_sku) # There was no SKU that was common to every STORE


# In[27]:


unique_SKU_per_store = pd.DataFrame(filtered_trnsact.groupby("STORE")["SKU"].nunique())


# In[28]:


dividing_with = 1000
lister = [dividing_with*x for x in list(range(0, int(max(unique_SKU_per_store["SKU"])/dividing_with)+2))]


# In[29]:


plt.rcParams['figure.figsize'] = [10, 5]
unique_SKU_per_store["SKU"].hist(bins = lister)
plt.title("Distribution of number of unique SKUs that each store has")
plt.xlabel("number of unique SKUs")
plt.ylabel("number of STOREs")
plt.savefig('img/Distribution of number of unique SKUs that each store has.png')
plt.show()


# In[30]:


filtered_store = unique_SKU_per_store[unique_SKU_per_store["SKU"] >= 18000]
filtered_store = filtered_store[filtered_store["SKU"] <= 19000]
filtered_store.head()


# In[31]:


filtered_trnsact = filtered_trnsact[filtered_trnsact["STORE"].isin(list(filtered_store.index))].reset_index(drop = True)


# ### 1.3.4. SKU Selection

# In[32]:


common_sku = filtered_trnsact[filtered_trnsact["STORE"] == min(filtered_trnsact["STORE"].unique())]["SKU"].unique()
for i in filtered_trnsact["STORE"].unique():
    cur = filtered_trnsact[filtered_trnsact["STORE"]==i]["SKU"].unique()
    common_sku = set(common_sku) & set(cur)
len(common_sku) # 343 unique SKUs shared among 25 STOREs


# In[33]:


filtered_trnsact = filtered_trnsact[filtered_trnsact["SKU"].isin(list(common_sku))]
filtered_trnsact["trnsact_identifier"].nunique()


# In[34]:


filtered_trnsact.to_csv("data/Dillards POS/trnsact_labelled_filtered.csv", index = False)


# ## <i> 1.4. Verifying whether our filtered data is representative enough </i>

# In[118]:


filtered_trnsact.to_csv("data/Dillards POS/trnsact_labelled_filtered.csv", index = False)


# In[119]:


fulfilling_store = 0
fulfilling_store_list = []
for index, row in pd.DataFrame(trnsact.groupby("STORE")["SKU"].unique()).iterrows():
    checker = sum(~pd.DataFrame(filtered_trnsact["SKU"].unique())[0].isin(list(row["SKU"])))
    if checker == 0: 
        fulfilling_store += 1   
        fulfilling_store_list.append(index)
print("%3.2f%% share these unique SKUs" % (100.0*fulfilling_store/(trnsact["STORE"].nunique())))


# ## <i> 1.5. Filtering other tables </i>
# ### 1.5.1 STRINFO

# In[120]:


strinfo = pd.read_csv("data/Dillards POS/strinfo_labelled.csv")
filtered_strinfo = strinfo[strinfo["STORE"].isin(list(filtered_store.index))]
print("Initial nrow: %i --> Filtered nrow: %i" % (strinfo.shape[0], filtered_strinfo.shape[0]))
filtered_strinfo.to_csv("data/Dillards POS/strinfo_labelled_filtered.csv", index = False)


# ### 1.5.2 SKSTINFO

# In[121]:


skstinfo = pd.read_csv("data/Dillards POS/skstinfo_labelled.csv")
filtered_skstinfo = skstinfo[skstinfo["STORE"].isin(list(filtered_store.index))]
filtered_skstinfo = filtered_skstinfo[filtered_skstinfo["SKU"].isin(list(filtered_trnsact["SKU"].unique()))]
print("Initial nrow: %i --> Filtered nrow: %i" % (skstinfo.shape[0], filtered_skstinfo.shape[0]))
filtered_skstinfo.to_csv("data/Dillards POS/skstinfo_labelled_filtered.csv", index = False)


# ### 1.5.3 SKUINFO & DEPTINFO

# In[122]:


skuinfo = pd.read_csv("data/Dillards POS/skuinfo_labelled.csv")
deptinfo = pd.read_csv("data/Dillards POS/deptinfo_labelled.csv")
sku_deptinfo = pd.merge(skuinfo, deptinfo, on = 'DEPT')
filtered_sku_deptinfo = sku_deptinfo[sku_deptinfo["SKU"].isin(list(filtered_trnsact["SKU"].unique()))]
print("Initial nrow: %i --> Filtered nrow: %i" % (sku_deptinfo.shape[0], filtered_sku_deptinfo.shape[0]))
filtered_sku_deptinfo.to_csv("data/Dillards POS/sku_deptinfo_labelled_filtered.csv", index = False)


# ---------
# # <u> 2. Exploratory Data Analysis (EDA)</u>
# ## <i> 2.1. Rechecking the structure of data </i>
# Since we have gone through filtering processes, I tried to recheck our data and confirm that there was no clear problem caused by the filtering. 
# ### 2.1.1 TRNSACT

# In[123]:


filtered_trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled_filtered.csv")
structure_checker(filtered_trnsact)


# ### 2.1.2. STRINFO

# In[124]:


filtered_strinfo = pd.read_csv("data/Dillards POS/strinfo_labelled_filtered.csv")
structure_checker(filtered_strinfo)


# ### 2.1.3. SKSTINFO

# In[125]:


filtered_skstinfo = pd.read_csv("data/Dillards POS/skstinfo_labelled_filtered.csv")
structure_checker(filtered_skstinfo)


# ### 2.1.4. SKUINFO & DEPTINFO

# In[126]:


filtered_sku_deptinfo = pd.read_csv("data/Dillards POS/sku_deptinfo_labelled_filtered.csv")
structure_checker(filtered_sku_deptinfo)


# ## <i> 2.2. Checking the relationship between data tables</i>

# In[127]:


def equal_checker(data1, data2, checking_column):
    list1 = data1.drop_duplicates(subset = checking_column)[checking_column].sort_values(by = checking_column).reset_index(drop = True)
    list2 = data2.drop_duplicates(subset = checking_column)[checking_column].sort_values(by = checking_column).reset_index(drop = True)
    return list1.equals(list2)


# ### 2.2.1. TRNSACT and SKSTINFO
# <b> SKU, STORE pair </b>

# In[128]:


equal_checker(filtered_trnsact, filtered_skstinfo, ["SKU", "STORE"])


# In[129]:


filtered_trnsact.drop_duplicates(subset = ["SKU", "STORE"]).shape[0]


# In[130]:


filtered_skstinfo.drop_duplicates(subset = ["SKU", "STORE"]).shape[0]


# <b> SKU </b>

# In[131]:


equal_checker(filtered_trnsact, filtered_skstinfo, ["SKU"])


# In[132]:


filtered_trnsact["SKU"].nunique()


# In[133]:


filtered_skstinfo["SKU"].nunique()


# <b> STORE </b>

# In[134]:


equal_checker(filtered_trnsact, filtered_skstinfo, ["STORE"])


# ### 2.2.2. STRINFO and SKSTINFO
# <b> STORE </b>

# In[135]:


equal_checker(filtered_strinfo, filtered_skstinfo, ["STORE"])


# ### 2.2.3. TRNSACT and SKU_DEPTINFO
# <b> SKU </b>

# In[136]:


equal_checker(filtered_trnsact, filtered_sku_deptinfo, ["SKU"])


# ### 2.2.4. TRNSACT and STRINFO
# <b> STORE </b>

# In[137]:


equal_checker(filtered_trnsact, filtered_strinfo, ["STORE"])


# ------

# # <u> 3. Applying the Apriori Algorithm</u>

# In[138]:


filtered_trnsact = pd.read_csv("data/Dillards POS/trnsact_labelled_filtered.csv")
pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[139]:


SKU_bool = pd.get_dummies(filtered_trnsact["SKU"], prefix='SKU')
SKU_bool["trnsact_identifier"] = list(filtered_trnsact["trnsact_identifier"])

for column in SKU_bool.columns:
    if column != "trnsact_identifier":
        SKU_bool[column] = SKU_bool[column].astype(bool)
SKU_bool.head()


# In[140]:


ForApriori = SKU_bool.groupby(["trnsact_identifier"]).any()
ForApriori.to_csv("data/ForApriori.csv", index = False)
ForApriori.head()


# In[141]:


FreqItems = apriori(ForApriori, min_support = 0.0005, use_colnames=True)
FreqItems.to_csv("data/FreqItems.csv", index = False)
FreqItems


# In[142]:


AssocRules = association_rules(FreqItems, metric ="lift", min_threshold = 1.5).sort_values("lift", ascending = False).reset_index(drop = True)
AssocRules


# In[143]:


AssocRules = AssocRules[AssocRules["confidence"] >= 0.01]


# In[144]:


AssocRules.to_csv("data/AssocRules.csv", index = False)


# ------

# # <u> 4. Insight & Discussion</u>
# ## <i> 4.1. Interpreting Basic Features of Created Rules</i>

# In[145]:


structure_checker(AssocRules)


# ## <i> 4.2. Identifying what SKUs to Move & where to move those SKUs</i>

# In[146]:


filtered_sku_deptinfo = pd.read_csv("data/Dillards POS/sku_deptinfo_labelled_filtered.csv")
filtered_sku_deptinfo.head()


# In[147]:


counter = 0 
pos_difference = [] # saves SKUs where CLASSID or DEPT do not match and CLASSID related to thsoe SKUs. 
for index, row in AssocRules.iterrows():
    antecedents = list(row["antecedents"]) 
    consequents = list(row["consequents"])
    total = antecedents+consequents
    total_cleaned = [int(x.replace("SKU_", "")) for x in total]
    if filtered_sku_deptinfo[filtered_sku_deptinfo["SKU"].isin(total_cleaned)]["DEPT"].nunique() != 1 :
        print("different DEPT")
    if filtered_sku_deptinfo[filtered_sku_deptinfo["SKU"].isin(total_cleaned)]["CLASSID"].nunique() != 1 :
        counter += 1
        SKU_loc = [] 
        for each_sku in total_cleaned:  
            SKU_loc.append(filtered_sku_deptinfo[filtered_sku_deptinfo["SKU"] == each_sku]["CLASSID"].iloc[0])
        pos_difference.append([total_cleaned, SKU_loc])
print(str(counter)+ " cases where products in antecedents and consequents were not in the same floor")


# In[148]:


pd.DataFrame(pos_difference)


# In[149]:


which_SKU_to_WHERE = [] 
for pos, val in enumerate(pos_difference): 
    which_SKU_to_WHERE.append([val[0][0], val[1][1]])
    which_SKU_to_WHERE.append([val[0][1], val[1][0]])


# In[150]:


which_SKU_to_WHERE = pd.DataFrame(which_SKU_to_WHERE)
which_SKU_to_WHERE.columns = ["SKU", "TO WHERE"]
which_SKU_to_WHERE = pd.DataFrame(which_SKU_to_WHERE.groupby(["SKU", "TO WHERE"]).size())
del which_SKU_to_WHERE[0]
which_SKU_to_WHERE = which_SKU_to_WHERE.reset_index()
which_SKU_to_WHERE


# ## <i> 4.3. Finding Patterns among SKUs that were recommended for Rearrangements</i>

# In[151]:


which_SKU_to_WHERE = which_SKU_to_WHERE.merge(filtered_sku_deptinfo, how = "left", left_on ="SKU", right_on= "SKU")
which_SKU_to_WHERE.to_csv("data/which_SKU_to_WHERE.csv")
which_SKU_to_WHERE.head(35)


# ## <i> 4.4. Validating identified SKU movement plan with other STOREs</i>

# In[152]:


which_SKU_to_WHERE = pd.read_csv("data/which_SKU_to_WHERE.csv")
which_SKU_to_WHERE.reset_index()

lister = [] 
for index, row in which_SKU_to_WHERE.iterrows():
    lister.append(str([row["SKU"], row["TO WHERE"]]))


# In[153]:


filtered_trnsact1 = pd.read_csv("data/Dillards POS/trnsact_labelled_filtered1.csv")
filtered_trnsact1 = filtered_trnsact1[filtered_trnsact1["SKU"].isin(filtered_trnsact["SKU"].unique())]


# In[154]:


def iterating_with_other_store(store_num, in_this_subset_question_mark, filtered_trnsact1): 
    filtered_trnsact1 = filtered_trnsact1[filtered_trnsact1["STORE"] == store_num]
    SKU_bool = pd.get_dummies(filtered_trnsact1["SKU"], prefix = 'SKU')
    SKU_bool["trnsact_identifier"] = list(filtered_trnsact1["trnsact_identifier"])

    for column in SKU_bool.columns:
        if column != "trnsact_identifier":
            SKU_bool[column] = SKU_bool[column].astype(bool)

    ForApriori = SKU_bool.groupby(["trnsact_identifier"]).any()
    FreqItems = apriori(ForApriori, min_support = 0.0005, use_colnames=True)

    AssocRules = association_rules(FreqItems, metric ="lift", min_threshold = 1.5).sort_values("lift", ascending = False).reset_index(drop = True)
    AssocRules = AssocRules[AssocRules["confidence"] >= 0.01]

    counter = 0 
    pos_difference = [] # saves SKUs where CLASSID or DEPT do not match and CLASSID related to thsoe SKUs. 
    for index, row in AssocRules.iterrows():
        antecedents = list(row["antecedents"]) 
        consequents = list(row["consequents"])
        total = antecedents+consequents
        total_cleaned = [int(x.replace("SKU_", "")) for x in total]
        if filtered_sku_deptinfo[filtered_sku_deptinfo["SKU"].isin(total_cleaned)]["CLASSID"].nunique() != 1 :
            counter += 1
            SKU_loc = [] 
            for each_sku in total_cleaned:  
                SKU_loc.append(filtered_sku_deptinfo[filtered_sku_deptinfo["SKU"] == each_sku]["CLASSID"].iloc[0])
            pos_difference.append([total_cleaned, SKU_loc])

    which_SKU_to_WHERE = [] 
    for pos, val in enumerate(pos_difference): 
        which_SKU_to_WHERE.append([val[0][0], val[1][1]])
        which_SKU_to_WHERE.append([val[0][1], val[1][0]])

    which_SKU_to_WHERE = pd.DataFrame(which_SKU_to_WHERE)
    which_SKU_to_WHERE.columns = ["SKU", "TO WHERE"]
    which_SKU_to_WHERE.head()

    which_SKU_to_WHERE = pd.DataFrame(which_SKU_to_WHERE.groupby(["SKU", "TO WHERE"]).size())
    del which_SKU_to_WHERE[0]
    which_SKU_to_WHERE = which_SKU_to_WHERE.reset_index()
    
    lister = [] 
    for index, row in which_SKU_to_WHERE.iterrows():
        lister.append(str([row["SKU"], row["TO WHERE"]]))
        
    return (100.0*sum(pd.DataFrame(lister)[0].isin(in_this_subset_question_mark))/len(in_this_subset_question_mark))


# In[155]:


random.seed(123)
probability = []
for trial in range(1, 21):
    print(trial)
    probability.append(iterating_with_other_store(fulfilling_store_list[trial], lister, filtered_trnsact1))


# In[156]:


plt.hist(probability)
plt.title("How much does each STORE support my proposed SKU rearrangement plan?")
plt.xlabel("% of support")
plt.ylabel("number of STOREs")
plt.savefig('img/How much does each STORE support my proposed SKU rearrangement plan?.png')
plt.show()

