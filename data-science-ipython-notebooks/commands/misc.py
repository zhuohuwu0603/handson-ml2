#!/usr/bin/env python
# coding: utf-8

# This notebook was prepared by [Donne Martin](http://donnemartin.com). Source and license info is on [GitHub](https://github.com/donnemartin/data-science-ipython-notebooks).

# # Misc Commands
# 
# * Anaconda
# * IPython Notebook
# * Git
# * Ruby
# * Jekyll
# * Pelican
# * Django

# <h2 id="anaconda">Anaconda</h2>

# [Anaconda](https://store.continuum.io/cshop/anaconda/) is a scientific python distribution containing Python, NumPy, SciPy, Pandas, IPython, Matplotlib, Numba, Blaze, Bokeh, and other great Python data analysis tools.

# In[ ]:


# See Anaconda installed packages
# get_ipython().system('conda list')

# List environments
# get_ipython().system('conda info -e')

# Create Python 3 environment
# get_ipython().system('conda create -n py3k python=3 anaconda')

# Activate Python 3 environment
# get_ipython().system('source activate py3k')

# Deactivate Python 3 environment
# get_ipython().system('source deactivate')

# Update Anaconda
# get_ipython().system('conda update conda')

# Update a package with Anaconda
# get_ipython().system('conda update ipython')

# Update a package
# get_ipython().system('conda update scipy')

# Update all packages
# get_ipython().system('conda update all')

# Install specific version of a package
# get_ipython().system('conda install scipy=0.12.0')

# Cleanup: Conda can accumulate a lot of disk space
# because it doesn’t remove old unused packages
# get_ipython().system('conda clean -p')

# Cleanup tarballs which are kept for caching purposes
# get_ipython().system('conda clean -t')


# <h2 id="ipython-notebook">IPython Notebook</h2>

# [IPython Notebook](http://ipython.org/notebook.html) is a "web-based interactive computational environment where you can combine code execution, text, mathematics, plots and rich media into a single document."

# In[ ]:


# Start IPython Notebook
ipython notebook

# Start IPython Notebook with built-in mode to work cleanly 
# with matplotlib figures
ipython notebook --pylab inline

# Start IPython Notebook with a profile
ipython notebook --profile=dark-bg

# Load the contents of a file
# get_ipython().run_line_magic('load', 'dir/file.py')

# Time execution of a Python statement or expression
# get_ipython().run_line_magic('timeit', '')
%%time

# Activate the interactive debugger
# get_ipython().run_line_magic('debug', '')

# Write the contents of the cell to a file
# get_ipython().run_line_magic('writefile', '')

# Run a cell via a shell command
%%script

# Run cells with bash in a subprocess
# This is a shortcut for %%script bash
%%bash

# Run cells with python2 in a subprocess
%%python2

# Run cells with python3 in a subprocess
%%python3

# Convert a notebook to a basic HTML file 
# get_ipython().system('ipython nbconvert --to html --template basic file.ipynb ')


# | Command   | Description                              |
# |-----------|------------------------------------------|
# | ?         | Intro and overview of IPython's features |
# | %quickref | Quick reference                          |
# | help      | Python help                              |
# | object?   | Object details, also use object??        |

# Apply css styling based on a css file:

# In[ ]:


from IPython.core.display import HTML

def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()


# <h2 id="git">Git</h2>

# [Git](http://git-scm.com/) is a distributed revision control system.

# In[ ]:


# Configure git
# get_ipython().system("git config --global user.name 'First Last'")
# get_ipython().system("git config --global user.email 'name@domain.com'")
# get_ipython().system('git init')

# View status and log
# get_ipython().system('git status')
# get_ipython().system('git log')

# Add or remove from staging area
# get_ipython().system('git add [target]')
# get_ipython().system('git reset [target file or commit]')
# get_ipython().system('git reset --hard origin/master')

# Automatically stage tracked files, 
# including deleting the previously tracked files
# Does not add untracked files
# get_ipython().system('git add -u')

# Delete files and stage them
# get_ipython().system('git rm [target]')

# Commit
# get_ipython().system('git commit -m “Add commit message here”')

# Add new origin
# get_ipython().system('git remote add origin https://github.com/donnemartin/ipython-data-notebooks.git')

# Set to new origin
# get_ipython().system('git remote set-url origin https://github.com/donnemartin/pydatasnippets.git')
    
# Push to master, -u saves config so you can just do "git push" afterwards
# get_ipython().system('git push -u origin master')
# get_ipython().system('git push')

# Diff files
# get_ipython().system('git diff HEAD')
# get_ipython().system('git diff --staged')
# get_ipython().system('git diff --cached')

# Show log message of commit and diff
# get_ipython().system('git show $COMMIT')

# Undo a file that has not been added
# get_ipython().system('git checkout — [target]')

# Revert a commit
# get_ipython().system('git revert')

# Undo a push and leave local repo intact
# get_ipython().system('git push -f origin HEAD^:master')

# Undo commit but leave files and index
# get_ipython().system('git reset --soft HEAD~1')

# Amend commit message of most recent change
# get_ipython().system('git commit --amend')
# get_ipython().system('git push --force [branch]')

# Take the dirty state of your working directory
# and save it on a stack of unfinished changes
# get_ipython().system('git stash')

# Get list of stashes
# get_ipython().system('git stash list')

# Apply the top stash, re-modifying the 
# uncommitted files when the stash was saved
# get_ipython().system('git stash apply')

# Apply a stash at the specified index
# get_ipython().system('git stash apply stash@{1}')

# Create a branch
# get_ipython().system('git branch [branch]')

# Check branches
# get_ipython().system('git branch')

# Switch branches
# get_ipython().system('git checkout [branch]')

# Merge branch to master
# get_ipython().system('git merge [branch]')

# Delete branch
# get_ipython().system('git branch -d [branch]')

# Clone
# get_ipython().system('git clone git@github.com:repo folder-name')
# get_ipython().system('git clone https://donnemartin@bitbucket.org/donnemartin/tutorial.git')
    
# Update a local repository with changes from a remote repository
# (pull down from master)
# get_ipython().system('git pull origin master')

# Configuring a remote for a fork
# get_ipython().system('git remote add upstream [target]')

# Set remote upstream
git branch --set-upstream-to origin/branch

# Check remotes
# get_ipython().system('git remote -v')

# Syncing a fork
# get_ipython().system('git fetch upstream')
# get_ipython().system('git checkout master')
# get_ipython().system('git merge upstream/master')

# Create a file containing a patch
# git format-patch are like normal patch files, but they also carry information 
# about the git commit that created the patch: the author, the date, and the 
# commit log message are all there at the top of the patch.
# get_ipython().system('git format-patch origin/master')

# Clean up .git folder:
# get_ipython().system('git repack -a -d --depth=250 --window=250')

# GitHub tutorial:
http://try.github.io/levels/1/challenges/9

# BitBucket Setup
# get_ipython().system('cd /path/to/my/repo')
# get_ipython().system('git init')
# get_ipython().system('git remote add origin https://donnemartin@bitbucket.org/donnemartin/repo.git')
# get_ipython().system('git push -u origin --all # pushes up the repo and its refs for the first time')
# get_ipython().system('git push -u origin --tags # pushes up any tags')

# Open Hatch missions
# get_ipython().system('git clone https://openhatch.org/git-mission-data/git/dmartin git_missions')


# <h2 id="ruby">Ruby</h2>

# [Ruby](https://www.ruby-lang.org/en/) is used to interact with the AWS command line and for Jekyll, a blog framework that can be hosted on GitHub Pages.

# In[ ]:


# Update Ruby
# get_ipython().system('rvm get stable')

# Reload Ruby (or open a new terminal)
# get_ipython().system('rvm reload')

# List all known RVM installable rubies
# get_ipython().system('rvm list known')

# List all installed Ruby versions
# get_ipython().system('rvm list')

# Install a specific Ruby version
# get_ipython().system('rvm install 2.1.5')

# Set Ruby version
# get_ipython().system('rvm --default ruby-1.8.7')
# get_ipython().system('rvm --default ruby-2.1.5')

# Check Ruby version
# get_ipython().system('ruby -v')


# <h2 id="jekyll">Jekyll</h2>

# [Jekyll](http://jekyllrb.com/) is a blog framework that can be hosted on GitHub Pages.
# 
# In addition to donnemartin.com, I’ve started to build up its mirror site donnemartin.github.io to try out Jekyll. So far I love that I can use my existing developer tools to generate content (SublimeText, Terminal, and GitHub).
# 
# Here are other features I like about Jekyll:
# 
# * Converts Markdown to produce fast, static pages
# * Simple to get started, no backend or manual updates
# * Hosted on GitHub Pages
# * Open source on GitHub

# Many Jekyll themes require a Ruby version of 2 and above.  However, the AWS CLI requires Ruby 1.8.7.  Run the proper version of Ruby for Jekyll:

# In[ ]:


# get_ipython().system('rvm --default ruby-2.1.5')


# Build and run the localy Jekyll server:

# In[ ]:


# => The current folder will be generated into ./_site
# get_ipython().system('bundle exec jekyll build')

# => A development server will run at http://localhost:4000/
# Auto-regeneration: enabled. Use `--no-watch` to disable.
# get_ipython().system('bundle exec jekyll serve')


# <h2 id="pelican">Pelican</h2>

# I've switched my personal website [donnemartin.com](http://donnemartin.com/) to run off Pelican, a python-based alternative to Jekyll.  Previous iterations ran off Wordpress and Jekyll.
# 
# Setup [reference](http://nafiulis.me/making-a-static-blog-with-pelican.html).

# In[ ]:


# Install
# get_ipython().system('pip install pelican')
# get_ipython().system('pip install markdown')
# get_ipython().system('pip install ghp-import')

# Quick retup
# get_ipython().system('pelican-quickstart')

# Run server
# get_ipython().system('make devserver')

# Stop server
# get_ipython().system('make stopserver')

# Run ghp-import on output folder
# Review https://pypi.python.org/pypi/ghp-import
# There's a "Big Fat Warning" section
# get_ipython().system('ghp-import output')

# Update gh-pages (if using a project page)
# get_ipython().system('git push origin gh-pages')

# Update gh-pages (if using a user or org page)
# get_ipython().system('git merge gh-pages master')


# ## Django

# [Django](https://www.djangoproject.com) is a high-level Python Web framework that encourages rapid development and clean, pragmatic design.  It can be useful to share reports/analyses and for blogging. Lighter-weight alternatives include [Pyramid](https://github.com/Pylons/pyramid), [Flask](https://github.com/mitsuhiko/flask), [Tornado](https://github.com/tornadoweb/tornado), and [Bottle](https://github.com/bottlepy/bottle).

# In[ ]:


# Check version of Django
# get_ipython().system('python -c "import django; print(django.get_version())"')

# Create and setup a project
# get_ipython().system('django-admin startproject mysite')

# Sync db
# get_ipython().system('python manage.py syncdb')

# The migrate command looks at the INSTALLED_APPS setting and 
# creates any necessary database tables according to the database 
# settings in your mysite/settings.py file and the database 
# migrations shipped with the app
# get_ipython().system('python manage.py migrate')

# Run the dev server
# get_ipython().system('python manage.py runserver')
1python manage.py runserver 8080
# get_ipython().system('python manage.py runserver 0.0.0.0:8000')

# Create app
# get_ipython().system('python manage.py startapp [app_label]')

# Run tests
python manage.py test [app_label]

# Tell Django that you’ve made some changes to your models 
# and that you’d like the changes to be stored as a migration.
# get_ipython().system('python manage.py makemigrations [app_label]')

# Take migration names and returns their SQL
# get_ipython().system('python manage.py sqlmigrate [app_label] [migration_number]')

# Checks for any problems in your project without making 
# migrations or touching the database.
# get_ipython().system('python manage.py check')

# Create a user who can login to the admin site
# get_ipython().system('python manage.py createsuperuser')

# Locate Django source files
# get_ipython().system('python -c "')
import sys
sys.path = sys.path[1:]
import django
print(django.__path__)"

