#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # PDB
# 
# The pdb module defines an interactive source code debugger for Python programs.  Below are frequently used commands:

# In[ ]:


# Run pdb when this line is hit
import pdb; pdb.set_trace()

# Run pdb when the script is run
python -m pdb script.py

# Help
h[elp]

# Show current content
l[ist]

# Examine variables
p[rint]

# Pretty print
pp

# Go to next line
n[ext]

# Step into
s[tep]

# Continue execution until the line with the line number greater 
# than the current one is reached or when returning from current frame.
until

# Return
r[eturn]

# See all breakpoints
b to see all breakpoints

# Set breakpoint at line 16
b 16 

# Clear breakpoint 1
cl[ear] 1

# Continue
c[ontinue]

# Conditional breakpoints, line 11
b 11, this_year == 2015

# Stack location
w[here]

# Go up in stack
u[p]

# Go down in stack
d[own]

# Longlist shows full method of where you're in (Python 3)
ll

# Quit
q[uit]

