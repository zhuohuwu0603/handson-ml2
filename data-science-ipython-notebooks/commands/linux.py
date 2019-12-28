#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Linux Commands
# 
# * Disk Usage
# * Splitting Files
# * Grep, Sed
# * Compression
# * Curl
# * View Running Processes
# * Terminal Syntax Highlighting
# * Vim

# ## Disk Usage

# Display human-readable (-h) free disk space:

# In[ ]:


# get_ipython().system('df -h')


# Display human-readable (-h) disk usage statistics:

# In[ ]:


# get_ipython().system('du -h ./')


# Display human-readable (-h) disk usage statistics, showing only the total usage (-s):

# In[ ]:


# get_ipython().system('du -sh ../')


# Display the human-readable (-h) disk usage statistics, showing also the grand total for all file types (-c):

# In[ ]:


# get_ipython().system('du -csh ./')


# ## Splitting Files

# Count number of lines in a file with wc:

# In[ ]:


# get_ipython().system('wc -l < file.txt')


# Count the number of lines in a file with grep:

# In[ ]:


# get_ipython().system('grep -c "." file.txt')


# Split a file into multiple files based on line count:

# In[ ]:


# get_ipython().system('split -l 20 file.txt new')


# Split a file into multiple files based on line count, use suffix of length 1:

# In[ ]:


# get_ipython().system('split -l 802 -a 1 file.csv dir/part-user-csv.tbl-')


# ## Grep, Sed

# List number of files matching “.txt":

# In[ ]:


# get_ipython().system('ls -1 | grep .txt | wc -l')


# Check number of MapReduce records processed, outputting the results to the terminal:

# In[ ]:


# get_ipython().system('cat * | grep -c "foo" folder/part*')


# Delete matching lines in place:

# In[ ]:


# get_ipython().system("sed -i '/Important Lines: /d’ original_file")


# ## Compression

# In[ ]:


# Compress zip
# get_ipython().system('zip -r archive_name.zip folder_to_compress')

# Compress zip without invisible Mac resources
# get_ipython().system('zip -r -X archive_name.zip folder_to_compress')

# Extract zip
# get_ipython().system('unzip archive_name.zip')

# Compress TAR.GZ
# get_ipython().system('tar -zcvf archive_name.tar.gz folder_to_compress')

# Extract TAR.GZ
# get_ipython().system('tar -zxvf archive_name.tar.gz')

# Compress TAR.BZ2
# get_ipython().system('tar -jcvf archive_name.tar.bz2 folder_to_compress')

# Extract TAR.BZ2
# get_ipython().system('tar -jxvf archive_name.tar.bz2')

# Extract GZ
# get_ipython().system('gunzip archivename.gz')

# Uncompress all tar.gz in current directory to another directory
# get_ipython().system('for i in *.tar.gz; do echo working on $i; tar xvzf $i -C directory/ ; done')


# ## Curl

# In[ ]:


# Display the curl output:
# get_ipython().system('curl donnemartin.com')

# Download the curl output to a file:
# get_ipython().system('curl donnemartin.com > donnemartin.html')

# Download the curl output to a file -o
# get_ipython().system('curl -o image.png http://i1.wp.com/donnemartin.com/wp-content/uploads/2015/02/splunk_cover.png')

# Download the curl output to a file, keeping the original file name -O
# get_ipython().system('curl -O http://i1.wp.com/donnemartin.com/wp-content/uploads/2015/02/splunk_cover.png')
    
# Download multiple files, attempting to reuse the same connection
# get_ipython().system('curl -O url1 -O url2')

# Follow redirects -L
# get_ipython().system('curl -L url')

# Resume a previous download -C -
# get_ipython().system('curl -C - -O url')

# Authenticate -u
# get_ipython().system('curl -u username:password url')


# ## View Running Processes

# In[ ]:


# Display sorted info about processes
# get_ipython().system('top')

# Display all running processes
# get_ipython().system('ps aux | less')

# Display all matching running processes with full formatting
# get_ipython().system('ps -ef | grep python')

# See processes run by user dmartin
# get_ipython().system('ps -u dmartin')

# Display running processes as a tree
# get_ipython().system('pstree')


# ## Terminal Syntax Highlighting

# Add the following to your ~/.bash_profile:

# In[ ]:


export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\$ '
export CLICOLOR=1
export LSCOLORS=ExFxBxDxCxegedabagacad
alias ls='ls -GFh'


# Reload .bash_profile:

# In[ ]:


# get_ipython().system('source ~/.bash_profile')


# ## Vim

# In[ ]:


Normal mode:  esc

Basic movement:  h, j, k, l
Word movement:  w, W, e, E, b, B

Go to matching parenthesis:  %
Go to start of the line:  0
Go to end of the line:  $

Find character:  f

Insert mode:  i
Append to line:  A

Delete character:  x
Delete command:  d
Delete line:  dd

Replace command:  r
Change command:  c

Undo:  u (U for all changes on a line)
Redo:  CTRL-R

Copy the current line:  yy
Paste the current line:  p (P for paste above cursor)

Quit without saving changes:  q!
Write the current file and quit:  :wq


# Run the following command to enable the tutorial:

# In[ ]:


# get_ipython().system('vimtutor')


# Run the following commands to enable syntax colors:

# In[ ]:


# get_ipython().system('cd ~')
# get_ipython().system('vim .vimrc')
# Add the following to ~/.vimrc
syntax on
:wq

