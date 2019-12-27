#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Dates and Times

# * Basics
# * strftime
# * strptime
# * timedelta

# ## Basics

# In[1]:


from datetime import datetime, date, time


# In[2]:


year = 2015
month = 1
day = 20
hour = 7
minute = 28
second = 15


# In[3]:


dt = datetime(year, month, day, hour, minute, second)


# In[4]:


dt.hour, dt.minute, dt.second


# Extract the equivalent date object:

# In[5]:


dt.date()


# Extract the equivalent time object:

# In[6]:


dt.time()


# When aggregating or grouping time series data, it is sometimes useful to replace fields of a series of datetimes such as zeroing out the minute and second fields:

# In[7]:


dt.replace(minute=0, second=0)


# ## strftime

# Format a datetime string:

# In[8]:


dt.strftime('%m/%d/%Y %H:%M')


# ## strptime

# Convert a string into a datetime object:

# In[9]:


datetime.strptime('20150120', '%Y%m%d')


# ## timedelta

# Get the current datetime:

# In[10]:


dt_now = datetime.now()


# Subtract two datetime fields to create a timedelta:

# In[11]:


delta = dt_now - dt
delta


# Add a datetime and a timedelta to get a new datetime:

# In[12]:


dt + delta

