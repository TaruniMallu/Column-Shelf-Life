#!/usr/bin/env python
# coding: utf-8


def main(df):
    # In[1]:

    #importing necessary Libraries
    
    import pandas as pd
    import sweetviz as sv
    import json


    # In[2]:


    from sklearn.feature_selection import SelectKBest, f_classif , mutual_info_classif
    from sklearn.preprocessing import LabelEncoder


    # In[3]:


    ##path = download('C:\\Users\\\\Downloads\\PeaksData 1.csv','tmp/aq',kind="zip")


    # In[4]:


    df = df


    # In[5]:


    df


    # In[6]:

    #preprocessing data

    df.isnull().values.any()


    # In[7]:


    null_columns = df.columns[df.isnull().all()]


    # In[8]:


    null_columns


    # In[9]:


    df_cleaned = df.drop(columns = null_columns)


    # In[10]:


    df_cleaned
    ## data after removing null columns


    # NUMERICAL COLUMNS ANALYSIS

    # In[12]:


    import numpy as np


    # In[13]:


    numeric_columns = df_cleaned.select_dtypes(include= [np.number]).columns


    # In[14]:


    numeric_columns


    # In[15]:


    df_cleaned[numeric_columns]=df_cleaned[numeric_columns].fillna(df_cleaned[numeric_columns].median())


    # In[16]:


    numeric_df = df_cleaned[numeric_columns]


    # In[17]:


    numeric_df
    ## data with only numerical columns


    # In[19]:


    #numeric_df.to_excel('numeric_data.xlsx',index = False, engine = 'xlsxwriter')


    # CATEGORICAL COLUMNS ANALYSIS

    # In[20]:


    categorical_columns = df_cleaned.select_dtypes(include = ['object']).columns


    # In[21]:


    categorical_columns


    # In[22]:


    df_cleaned[categorical_columns]=df_cleaned[categorical_columns].fillna(df_cleaned[categorical_columns].mode().iloc[0])


    # In[23]:


    categoric_df = df_cleaned[categorical_columns]


    # In[24]:


    categoric_df
    #df containing numeric columns


    # In[25]:


    ## chi-square test of independence( checks is there's a significant association between two categorical columns.)
    from scipy.stats import chi2_contingency


    # In[32]:


    # Function to conduct Chi-Square tests between all pairs of categorical columns
    def chi_square_tests(categoric_df):
        results = []
        for i, col1 in enumerate(categorical_columns):
            for col2 in categorical_columns[i+1:]:
                contingency_table = pd.crosstab(categoric_df[col1],categoric_df[col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                results.append((col1, col2, chi2, p,dof, expected))
        return results


    # In[33]:


    final_result = chi_square_tests(categoric_df)


    # In[34]:


    # Print the results
    for result in final_result:
        col1, col2, chi2, p, dof, expected = result
        print(f"Chi-Square Test between '{col1}' and '{col2}':")
        print(f"Chi2: {chi2}")
        print(f"P-value: {p}")
        print(f"Degrees of Freedom: {dof}")
        print(f"Expected Frequencies:\n{expected}")
        print()


    # In[35]:


    categorical_results_df = pd.DataFrame(final_result)


    # In[37]:


    #categorical_results_df.to_csv('chi_square_results.csv', index=False)


    # FEATURE SELECTION FOR NUMERICAL COLUMNS

    # In[ ]:


    # Identify potential target columns
    def identify_potential_targets(numeric_df):
        potential_targets = []
        for col in numeric_df.columns:
            if numeric_df[col].dtype == 'object' or numeric_df[col].nunique() < 10:
                potential_targets.append(col)
        return potential_targets

    potential_target = identify_potential_targets(numeric_df)
    print(potential_target)
    number_potential_target = len(potential_target)
    print('number of potential targets is: ',number_potential_target)


    # In[ ]:


    # Evaluate relationships and identify the best target column
    def identify_best_target(numeric_df, potential_targets):
        scores = {}
        for target in potential_targets:
            X = numeric_df.drop(columns=[target])
            y = LabelEncoder().fit_transform(numeric_df[target])
            
            mi = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
            scores[target] = mi.mean()
        
        best_target = max(scores, key=lambda key: scores[key])
        return best_target

    


    # In[ ]:


    # Feature selection
    def select_best_features(numeric_df, target, k=10):
        X = numeric_df.drop(columns=[target])
        y = LabelEncoder().fit_transform(numeric_df[target])
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        return selected_features


    # In[ ]:


    """import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)


    if not hasattr(np, 'VisibleDeprecationWarning'):
        class VisibleDeprecationWarning(Warning):
            pass
        np.VisibleDeprecationWarning = VisibleDeprecationWarning


    # In[ ]:


    # Generate EDA report
    def generate_eda_report(numeric_df, target):
        report = sv.analyze(numeric_df, target_feat=target, feat_cfg=sv.FeatureConfig(force_num=[target]))
        report.show_html('sweet_report.html')
        print("Sweetviz report generated successfully.")"""


    # In[ ]:


    # Main function to automate EDA and feature selection
    def automate_feature_selection(numeric_df):
        potential_targets = identify_potential_targets(numeric_df)
        best_target = identify_best_target(numeric_df, potential_targets)
        print("Best target column identified:", best_target)
        
        selected_features = select_best_features(numeric_df, best_target)
        print("Selected features:", selected_features)
        
        #generate_eda_report(numeric_df, best_target)
        
        return selected_features


    # In[ ]:


    # Run the automation
    selected_features = automate_feature_selection(numeric_df)
    print("Final selected features:", selected_features)
    
    # Create a dictionary with default values (0 in this example)
    result = {feature: 0 for feature in selected_features}
    
    #converting the dictionary into a json format
    json_result = json.dumps(result)


    """# In[ ]:


    # Function to identify the best related feature
    def identify_best_related_feature(numeric_df, selected_features, target):
        # Calculate correlation coefficients between selected features and target
        correlations = {}
        for feature in selected_features:
            corr = np.abs(np.corrcoef(numeric_df[feature], numeric_df[target])[0, 1])
            correlations[feature] = corr
        
        # Get the feature with the highest correlation coefficient
        best_related_feature = max(correlations.keys(), key=lambda k: correlations[k])
        
        return best_related_feature


    # In[ ]:


    target = 'RelativeResponse'  # Replace with your actual target column name
    best_related_feature = identify_best_related_feature(numeric_df, selected_features, target)


    # In[ ]:


    print("Selected features:", selected_features)
    print("Best related feature:", best_related_feature)


    # In[ ]:


    import pandas as pd

    # Assuming df is your DataFrame containing both columns
    correlation = df['USPTailing'].corr(df['InflectionWidth'])
    print("Correlation coefficient:", correlation)"""
    
    return json_result

