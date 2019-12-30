#!/usr/bin/env python
# coding: utf-8

# # T81-558: Applications of Deep Neural Networks
# **Module 12: Deep Learning and Security**
# * Instructor: [Jeff Heaton](https://sites.wustl.edu/jeffheaton/), McKelvey School of Engineering, [Washington University in St. Louis](https://engineering.wustl.edu/Programs/Pages/default.aspx)
# * For more information visit the [class website](https://sites.wustl.edu/jeffheaton/t81-558/).

# # Module 12 Video Material
# 
# * Part 12.1: Introduction to the OpenAI Gym [[Video]](https://www.youtube.com/watch?v=_KbUxgyisjM&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_01_ai_gym.ipynb)
# * Part 12.2: Introduction to Q-Learning [[Video]](https://www.youtube.com/watch?v=uwcXWe_Fra0&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_02_qlearningreinforcement.ipynb)
# * Part 12.3: Keras Q-Learning in the OpenAI Gym [[Video]](https://www.youtube.com/watch?v=Ya1gYt63o3M&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_03_keras_reinforce.ipynb)
# * Part 12.4: Atari Games with Keras Neural Networks [[Video]](https://www.youtube.com/watch?v=t2yIu6cRa38&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_04_atari.ipynb)
# * **Part 12.5: How Alpha Zero used Reinforcement Learning to Master Chess** [[Video]](https://www.youtube.com/watch?v=ikDgyD7nVI8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_05_alpha_zero.ipynb)
# 

# # Part 12.5: How Alpha Zero used Reinforcement Learning to Master Chess
# 
# ### AlphaGo
# 
# [Game of Go](https://en.wikipedia.org/wiki/Go_(game)) is:
# * 3,000 years old
# * 40M Players
# * $10^{170}$ board positions ($10^{78}$ to $10^{82}$ atoms in the known, observable universe)
# 
# 
# The original Google AlphaGo made use of two convolutional neural networks:
# 
# * **Policy Network** - CNN that accepts a Go Board (19x19) and decides on a next move. 
# * **Value Network** - CNN that accepts a Go Board (19x19) and evaluates who the likely winner will be.  Predicts winner or game -1 and +1 (win/loss).
# 
# ![AlphaGo Two Networks](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-1.png "AlphaGo Two Networks")
# 
# Supervised learning based on human games.  Reproduce (policy network) 
# Policy network plays itself
# 
# Train value network by reinforcement learning
# 
# ![AlphaGo Training Pipeline](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-2.png "AlphaGo Training Pipeline")
# 
# Narrow search space to reduce width of search
# Value network needs less depth
# 
# Monte-Carlo tree search
# 1. Traverse 
# 
# Lost one game.  Delusions
# 
# Random rollouts
# 
# ### Exhaustive Search
# 
# ![Exhaustive Search](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-4.png "Exhaustive Search")
# 
# ### Reducing Breadth with Policy Network
# 
# ![Reducing Breadth with Policy Network](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-5.png "Reducing Breadth with Policy Network")
# 
# ### Monte Carlo Tree Search (MCTS): Selection
# 
# [Monte Carlo tree search (MCTS)](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) is a heuristic search algorithm for some kinds of decision processes, most notably those employed in game play. MCTS was introduced in 2006 for computer Go. It has been used in other board games like chess and shogi, games with incomplete information such as bridge and poker, as well as in real-time video games.
# 
# MCTS consists of four steps:
# 
# * **Selection**: start from root R and select successive child nodes until a leaf node L is reached. The root is the current game state and a leaf is any node from which no simulation (playout) has yet been initiated. The section below says more about a way of biasing choice of child nodes that lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search.
# * **Expansion**: unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child nodes and choose node C from one of them. Child nodes are any valid moves from the game position defined by L.
# * **Simulation**: complete one random playout from node C. This step is sometimes also called playout or rollout. A playout may be as simple as choosing uniform random moves until the game is decided (for example in chess, the game is won, lost, or drawn).
# * **Backpropagation**: use the result of the playout to update information in the nodes on the path from C to R.
# 
# 
# * Q-Value is the accumulated rewards from previous move selection
# * U-Value ii policy network reward
# 
# The selection step of MCTS for AlphaGo is summarized as follows:
# 
# ![Monte Carlo Tree Search (MCTS): Selection](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-6.png "Monte Carlo Tree Search (MCTS): Selection")
# 
# ### Monte Carlo Tree Search (MCTS): Expansion
# 
# The expansion step of MCTS for AlphaGo is summarized as follows:
# 
# ![Monte Carlo Tree Search (MCTS): Expansion](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-7.png "Monte Carlo Tree Search (MCTS): Expansion")
# 
# ### Monte Carlo Tree Search (MCTS): Evaluation
# 
# The evaluation step of MCTS for AlphaGo is summarized as follows:
# 
# ![Monte Carlo Tree Search (MCTS): Evaluation](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-8.png "Monte Carlo Tree Search (MCTS): Evaluation")
# 
# ### Monte Carlo Tree Search (MCTS): Backup
# 
# The backup step of MCTS for AlphaGo is summarized as follows:
# 
# ![Monte Carlo Tree Search (MCTS): Backup](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-9.png "Monte Carlo Tree Search (MCTS): Backup")
# 
# ### AlphaGo Hardware
# 
# AlphaGo requires considerable hardware to traverse while performing MCTS.  Later versions will improve this.
# 
# ![AlphaGo Hardware](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-10.png "AlphaGo Hardware")
# 

# ## AlphaGo Master
# 
# AlphaGo master is similar to AlphaGo, except it does away with the breakout step.  AlphaGo Master played Ke Jia, AlphaGo Master Won 3-0.

# ### AlphaGo Zero
# 
# * No human data
# * No human feature engineering
# * Single neural network (combine policy and value networks)
# * Simpler search
# 
# Combined single network is made up of: many residual blocks 4 of convolutional layers 16, 17 with batch normalization 18 and rectifier non-linearities 19.
# 
# ### Reinforcement Learning
# 
# AlphaGo Zero is trained entirely by self-play from initially from policies that are quite random.  The combined neural network is trained at each iteration to predict the same move as the 
# 
# ![AlphaGo Zero Zero Plays Itself](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-11.png "AlphaGo Zero Plays Itself")
# 
# ![Policy Zero is Trained Against AlphaGo Zero's Moves](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-12.png "Reinforcement Learning in AlphaGo Master")
# 
# ![Value Network is Trained to Predict Winner](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-13.png "Value Network is Trained to Predict Winner")
# 
# ![New Value and Policy Network used in Next Iteration](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-14.png "New Value and Policy Network used in Next Iteration")
# 
# Two-step process:
# 
# * **Search-Based Policy Improvement**
#     * Run MCTS search using current network
#     * Actions selected by MCTS > actions selected by raw network
# * **Seach-Based Policy Evaluation**
#     * Play self-play games with AlphaGo search
#     * Evaluate improved policy by the average outcome
# 
# State of the art residual network
# 
# Random rollouts removed, only used neural network to evaluate.  More general, maybe apply to other games.
# 
# MCTS - Lookahead, train neural network (policy) to come up with same result.
# Train value net to better predict winner
# 
# Iterate over:
# * Search-Based Policy Improvement
# * Search-Based Policy Evaluation (key feature)
# 
# 
# AlphaGo Zero uses a single neural network to suggest moves, so it requires much less computation power than previous version.  
# 
# ![Alpha Go Master](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-15.png "AlphaGo Master Hardware")
# 

# ### AlphaZero
# 
# * chess
# * shogi
# * Go
# 
# ![New Value and Policy Network used in Next Iteration](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/alpha-zero-16.png "New Value and Policy Network used in Next Iteration")
# 
# Chess is:
# 
# * Most studied domain in the history of AI
# * Highly specialized systems have been successful in chess
# * Shogi (Japanese Chess) is computationally harder than chess
# * State of the art engines are based on alpha-beta search (form of minimax)
# 
# 2016 TCEC - Stockfish has very specialized algorithms for each of the following:
# 
# * Board Representation
# * Search
# * Transposition Table
# * Move Ordering
# * Selectivity
# * Evaluation
# * End Game Tablebases
# 
# Elo rating
# 
# 
# 
# 
# 

# In[ ]:




