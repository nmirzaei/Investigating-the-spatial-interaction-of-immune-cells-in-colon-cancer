import os
import platform
from subprocess import call

def check_dependencies():


  ###############################################################
  #Checking for pandas installation
  #If not ask user for permission to install
  ###############################################################
  try:
      import pandas as pd
      print('pandas installation: checked')
  except:
      answer1 = input('A pandas package installation is required for this code. Do you want to install it?(Yes=1, No=0)')
      if int(answer1)==0:
          print('Error: pandas is required for this code.\n')
      elif int(answer1)==1:
          print('**Starting pandas installation**')
          cmd6 = 'pip3 install --user pandas'
          os.system(cmd6)
      else:
          print('Error: Input not acceptable.')
          exit()
  ###############################################################

  ###############################################################
  #Checking for numpy installation
  #If not ask user for permission to install
  ###############################################################

  try:
      import numpy as np
      print('numpy installation: checked')
  except:
      answer1 = input('A numpy package installation is required for this code. Do you want to install it?(Yes=1, No=0)')
      if int(answer1)==0:
          print('Error: numpy is required for this code.\n')
      elif int(answer1)==1:
          print('**Starting numpy installation**')
          cmd7 = 'pip3 install --user numpy'
          os.system(cmd7)
      else:
          print('Error: Input not acceptable.')
          exit()
  ###############################################################
