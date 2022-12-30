from dolfin import *
import os
import platform
from subprocess import call

def check_dependencies():

  with open('Mesh1.geo', 'w') as the_file:
      the_file.close()
  ###############################################################
  #Checking for FEniCS compatible GMSH installation
  #If not ask user for permission to install
  ###############################################################
  try:
      call(["gmsh","-v", "0", "-2", 'Mesh1.geo'])
      print('FEniCS compatible GMSH installation: checked')
  except:
      answer = input('A FEniCS compatible GMSH installation is required for this code. Do you want to install it?(Yes=1, No=0)')
      if int(answer)==0:
          print('Warning: This code will not be able to do remeshing without a FEniCS compatible GMSH installation.\n')
          print('Warning: No remeshing might cause in domain instabilities.\n')
          print('**Either Choose no remeshing when prompted or rerun the code and install the FEniCS compatible GMSH installation.**')
      elif int(answer)==1:
          print('**Starting FEniCS compatible GMSH installation**')
          cmd1 = '/bin/sh -c "sudo apt-get update; sudo apt-get install -y libgl1-mesa-glx libxcursor1 libxft2 libxinerama1 libglu1-mesa"'
          cmd2 = 'wget -nc --quiet gmsh.info/bin/Linux/gmsh-3.0.6-Linux64.tgz'
          cmd3 = 'tar -xf gmsh-3.0.6-Linux64.tgz'
          cmd4 = 'sudo cp -r gmsh-3.0.6-Linux64/share/* /usr/local/share/'
          cmd5 = 'sudo cp gmsh-3.0.6-Linux64/bin/* /usr/local/bin'
          os.system(cmd1)
          os.system(cmd2)
          os.system(cmd3)
          os.system(cmd4)
          os.system(cmd5)
      else:
          print('Error: Input not acceptable.')
          exit()
  ###############################################################

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
