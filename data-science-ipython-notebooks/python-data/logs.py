#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Logging in Python
# * Logging with RotatingFileHandler
# * Logging with TimedRotatingFileHandler 

# ## Logging with RotatingFileHandler
# 
# The logging discussion is taken from the [Python Logging Cookbook](https://docs.python.org/2/howto/logging-cookbook.html#using-file-rotation):
# 
# Sometimes you want to let a log file grow to a certain size, then open a new file and log to that. You may want to keep a certain number of these files, and when that many files have been created, rotate the files so that the number of files and the size of the files both remain bounded. For this usage pattern, the logging package provides a RotatingFileHandler.
# 
# The most current file is always logging_rotatingfile_example.out, and each time it reaches the size limit it is renamed with the suffix .1. Each of the existing backup files is renamed to increment the suffix (.1 becomes .2, etc.) and the .6 file is erased.
# 
# The following code snippet is taken from [here](http://www.blog.pythonlibrary.org/2014/02/11/python-how-to-create-rotating-logs/).

# In[ ]:


import logging
import time
 
from logging.handlers import RotatingFileHandler
 
#----------------------------------------------------------------------
def create_rotating_log(path):
    """
    Creates a rotating log
    """
    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.INFO)
 
    # add a rotating handler
    handler = RotatingFileHandler(path, maxBytes=20,
                                  backupCount=5)
    logger.addHandler(handler)
 
    for i in range(10):
        logger.info("This is test log line %s" % i)
        time.sleep(1.5)
 
#----------------------------------------------------------------------
if __name__ == "__main__":
    log_file = "test.log"
    create_rotating_log(log_file)


# ## Logging with TimedRotatingFileHandler
# 
# The following code snippet is taken from [here](http://www.blog.pythonlibrary.org/2014/02/11/python-how-to-create-rotating-logs/).

# In[ ]:


import logging
import time
 
from logging.handlers import TimedRotatingFileHandler
 
#----------------------------------------------------------------------
def create_timed_rotating_log(path):
    """"""
    logger = logging.getLogger("Rotating Log")
    logger.setLevel(logging.INFO)
 
    # Rotate log based on when parameter:
    # second (s)
    # minute (m)
    # hour (h)
    # day (d)
    # w0-w6 (weekday, 0=Monday)
    # midnight
    handler = TimedRotatingFileHandler(path,
                                       when="m",
                                       interval=1,
                                       backupCount=5)
    logger.addHandler(handler)
 
    for i in range(20):
        logger.info("This is a test!")
        time.sleep(1.5)
 
#----------------------------------------------------------------------
if __name__ == "__main__":
    log_file = "timed_test.log"
    create_timed_rotating_log(log_file)

