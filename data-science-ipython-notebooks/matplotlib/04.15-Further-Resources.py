#!/usr/bin/env python
# coding: utf-8

# <!--BOOK_INFORMATION-->
# <img align="left" style="padding-right:10px;" src="figures/PDSH-cover-small.png">
# *This notebook contains an excerpt from the [Python Data Science Handbook](http://shop.oreilly.com/product/0636920034919.do) by Jake VanderPlas; the content is available [on GitHub](https://github.com/jakevdp/PythonDataScienceHandbook).*
# 
# *The text is released under the [CC-BY-NC-ND license](https://creativecommons.org/licenses/by-nc-nd/3.0/us/legalcode), and code is released under the [MIT license](https://opensource.org/licenses/MIT). If you find this content useful, please consider supporting the work by [buying the book](http://shop.oreilly.com/product/0636920034919.do)!*
# 
# *No changes were made to the contents of this notebook from the original.*

# <!--NAVIGATION-->
# < [Visualization with Seaborn](04.14-Visualization-With-Seaborn.ipynb) | [Contents](Index.ipynb) | [Machine Learning](05.00-Machine-Learning.ipynb) >

# # Further Resources

# ## Matplotlib Resources
# 
# A single chapter in a book can never hope to cover all the available features and plot types available in Matplotlib.
# As with other packages we've seen, liberal use of IPython's tab-completion and help functions (see [Help and Documentation in IPython](01.01-Help-And-Documentation.ipynb)) can be very helpful when exploring Matplotlib's API.
# In addition, Matplotlib’s [online documentation](http://matplotlib.org/) can be a helpful reference.
# See in particular the [Matplotlib gallery](http://matplotlib.org/gallery.html) linked on that page: it shows thumbnails of hundreds of different plot types, each one linked to a page with the Python code snippet used to generate it.
# In this way, you can visually inspect and learn about a wide range of different plotting styles and visualization techniques.
# 
# For a book-length treatment of Matplotlib, I would recommend [*Interactive Applications Using Matplotlib*](https://www.packtpub.com/application-development/interactive-applications-using-matplotlib), written by Matplotlib core developer Ben Root.

# ## Other Python Graphics Libraries
# 
# Although Matplotlib is the most prominent Python visualization library, there are other more modern tools that are worth exploring as well.
# I'll mention a few of them briefly here:
# 
# - [Bokeh](http://bokeh.pydata.org) is a JavaScript visualization library with a Python frontend that creates highly interactive visualizations capable of handling very large and/or streaming datasets. The Python front-end outputs a JSON data structure that can be interpreted by the Bokeh JS engine.
# - [Plotly](http://plot.ly) is the eponymous open source product of the Plotly company, and is similar in spirit to Bokeh. Because Plotly is the main product of a startup, it is receiving a high level of development effort. Use of the library is entirely free.
# - [Vispy](http://vispy.org/) is an actively developed project focused on dynamic visualizations of very large datasets. Because it is built to target OpenGL and make use of efficient graphics processors in your computer, it is able to render some quite large and stunning visualizations.
# - [Vega](https://vega.github.io/) and [Vega-Lite](https://vega.github.io/vega-lite) are declarative graphics representations, and are the product of years of research into the fundamental language of data visualization. The reference rendering implementation is JavaScript, but the API is language agnostic. There is a Python API under development in the [Altair](https://altair-viz.github.io/) package. Though as of summer 2016 it's not yet fully mature, I'm quite excited for the possibilities of this project to provide a common reference point for visualization in Python and other languages.
# 
# The visualization space in the Python community is very dynamic, and I fully expect this list to be out of date as soon as it is published.
# Keep an eye out for what's coming in the future!

# <!--NAVIGATION-->
# < [Visualization with Seaborn](04.14-Visualization-With-Seaborn.ipynb) | [Contents](Index.ipynb) | [Machine Learning](05.00-Machine-Learning.ipynb) >
