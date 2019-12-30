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
# * **Part 12.4: Atari Games with Keras Neural Networks** [[Video]](https://www.youtube.com/watch?v=t2yIu6cRa38&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_04_atari.ipynb)
# * Part 12.5: How Alpha Zero used Reinforcement Learning to Master Chess [[Video]](https://www.youtube.com/watch?v=ikDgyD7nVI8&list=PLjy4p-07OYzulelvJ5KVaT2pDlxivl_BN) [[Notebook]](t81_558_class_12_05_alpha_zero.ipynb)
# 

# # Part 12.4: Atari Games with Keras Neural Networks
# 
# 
# The Atari 2600 is a home video game console from Atari, Inc. Released on September 11, 1977. It is credited with popularizing the use of microprocessor-based hardware and games stored on ROM cartridges instead of dedicated hardware with games physically built into the unit. The 2600 was bundled with two joystick controllers, a conjoined pair of paddle controllers, and a game cartridge: initially [Combat](https://en.wikipedia.org/wiki/Combat_(Atari_2600)), and later [Pac-Man](https://en.wikipedia.org/wiki/Pac-Man_(Atari_2600)).
# 
# Atari emulators are popular and allow many of the old Atari video games to be played on modern computers.  They are even available as JavaScript.
# 
# * [Virtual Atari](http://www.virtualatari.org/listP.html)
# 
# Atari games have become popular benchmarks for AI systems, particularly reinforcement learning.  OpenAI Gym internally uses the [Stella Atari Emulator](https://stella-emu.github.io/).  
# 
# ![Atari 2600 Console](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/atari-1.png "Atari 2600 Console")
# 
# ### Installing Atari Emulator
# 
# ```
# pip install gym[atari]
# ```
# 
# ### Actual Atari 2600 Specs
# 
# * CPU: 1.19 MHz MOS Technology 6507
# * Audio + Video processor: Television Interface Adapter (TIA)
# * Playfield resolution: 40 x 192 pixels (NTSC). Uses a 20-pixel register that is mirrored or copied, left side to right side, to achieve the width of 40 pixels.
# * Player sprites: 8 x 192 pixels (NTSC). Player, ball, and missile sprites use pixels that are 1/4 the width of playfield pixels (unless stretched).
# * Ball and missile sprites: 1 x 192 pixels (NTSC).
# * Maximum resolution: 160 x 192 pixels (NTSC). Max resolution is only somewhat achievable with programming tricks that combine sprite pixels with playfield pixels.
# * 128 colors (NTSC). 128 possible on screen. Max of 4 per line: background, playfield, player0 sprite, and player1 sprite. Palette switching between lines is common. Palette switching mid line is possible but not common due to resource limitations.
# * 2 channels of 1-bit monaural sound with 4-bit volume control.
# 
# ### OpenAI Lab Atari Breakout
# 
# OpenAI Gym can be used with Windows; however, it requires a special [installation procedure](https://towardsdatascience.com/how-to-install-openai-gym-in-a-windows-environment-338969e24d30)  On Mac/Linux installation is as easy as:
# 
# ![Atari Breakout](https://raw.githubusercontent.com/jeffheaton/t81_558_deep_learning/master/images/atari-2.png "Atari Breakout")
# 
# (from Wikipedia)
# 
# Breakout begins with eight rows of bricks, with each two rows a different color. The color order from the bottom up is yellow, green, orange and red. Using a single ball, the player must knock down as many bricks as possible by using the walls and/or the paddle below to ricochet the ball against the bricks and eliminate them. If the player's paddle misses the ball's rebound, he or she will lose a turn. The player has three turns to try to clear two screens of bricks. Yellow bricks earn one point each, green bricks earn three points, orange bricks earn five points and the top-level red bricks score seven points each. The paddle shrinks to one-half its size after the ball has broken through the red row and hit the upper wall. Ball speed increases at specific intervals: after four hits, after twelve hits, and after making contact with the orange and red rows.
# 
# The highest score achievable for one player is 896; this is done by eliminating two screens of bricks worth 448 points per screen. Once the second screen of bricks is destroyed, the ball in play harmlessly bounces off empty walls until the player restarts the game, as no additional screens are provided. However, a secret way to score beyond the 896 maximum is to play the game in two-player mode. If "Player One" completes the first screen on his or her third and last ball, then immediately and deliberately allows the ball to "drain", Player One's second screen is transferred to "Player Two" as a third screen, allowing Player Two to score a maximum of 1,344 points if he or she is adept enough to keep the third ball in play that long. Once the third screen is eliminated, the game is over.
# 
# The original arcade cabinet of Breakout featured artwork that revealed the game's plot to be that of a prison escape. According to this release, the player is actually playing as one of a prison's inmates attempting to knock a ball and chain into a wall of their prison cell with a mallet. If the player successfully destroys the wall in-game, their inmate escapes with others following.
# 
# ### Breakout (BreakoutDeterministic-v4) Specs:
# 
# * BreakoutDeterministic-v4
# * State size (RGB): (210, 160, 3)
# * Actions: 4 (discrete)
# 
# The video for this course demonstrated playing Breakout.  The following [example code](https://github.com/wau/keras-rl2/blob/master/examples/dqn_atari.py) was used.

# The following code can be used to probe an environment to see the shape of its states and actions.

# In[10]:


import gym

env = gym.make("BreakoutDeterministic-v4")
print(f"Obesrvation space: {env.observation_space}")
print(f"Action space: {env.action_space}")


# In[ ]:




