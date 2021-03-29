#!/usr/bin/env python
# coding: utf-8

# In[1]:


#read params
# process data


# In[51]:


import os
import yaml
import pandas as pd
import numpy as np
import argparse

class process_data:
    
    def get_data_func(self, data_path):
        df = pd.read_csv(data_path, sep=',')
        return df
    
    def split_raw_data(self, df, config_path):
        cols = [col.replace(' ', '_') for col in df.columns]
        print(cols)
        df.to_csv(config_path, index=False, header=cols, sep=',')

    def __init__(self, path=''):
        if(path == ''):
            self.path = 'params.yaml'
        self.raw_data = dict()
        with open(self.path) as f:
            self.raw_data  = yaml.safe_load(f)
        print(self.raw_data)
        df = self.get_data_func(self.raw_data['data_source']['s3_source'])
        self.split_raw_data(df, self.raw_data['load_data']['raw_dataset_csv'])

if __name__ == "__main__":
	process_data()
		
# In[52]:


#process_data()


# In[44]:


#"a a".replace(" ", "_")


# In[ ]:




