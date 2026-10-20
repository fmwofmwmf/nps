import sys
import os
import numpy as np

# here are some colors of my color pallete
yellow = (242.0/255.0,122.0/255.0,42.0/255.0,1.0)
orange = (233.0/255.0,76.0/255.0,38.0/255.0,1.0)
blue = (58.0/255.0,119.0/255.0,157.0/255.0,1.0)
dark = (24.0/255.0,37.0/255.0,77.0/255.0,1.0)
black = (0.0/255.0,0.0/255.0,0.0/255.0,1.0)

# here are some colors from Cynthia Brewer's color palette.

blue_1 = (166.0/255.0,206.0/255.0,227.0/255.0,1.0)
blue_2 = (31.0/255.0,120.0/255.0,180.0/255.0,1.0)
green_1 = (178.0/255.0,223.0/255.0,138.0/255.0,1.0)
green_2 = (51.0/255.0,160.0/255.0,44.0/255.0,1.0)
red_1 = (251.0/255.0,154.0/255.0,153.0/255.0,1.0)
red_2 = (227.0/255.0,26.0/255.0,28.0/255.0,1.0)
orange_1 = (253.0/255.0,191.0/255.0,111.0/255.0,1.0)
orange_2 = (255.0/255.0,127.0/255.0,0.0/255.0,1.0)
purple_1 = (202.0/255.0,178.0/255.0,214.0/255.0,1.0)
purple_2 = (106.0/255.0,61.0/255.0,154.0/255.0,1.0)
yellow_1 = (255.0/255.0,255.0/255.0,153.0/255.0,1.0)
yellow_2 = (225.0/255.0,214.0/255.0,35.0/255.0,1.0)
brown = (177.0/255.0,89.0/255.0,40.0/255.0,1.0)
grey_1 = (128.0/255.0,128.0/255.0,128.0/255.0,1.0)
grey_2 = (45.0/255.0,45.0/255.0,45.0/255.0,1.0)
pink = (221/255.0,52/255.0,151/255.0,1.0)

# Here we assign meaning to the colors, that are very consistent. Throughout our WHOLE code, these are the only colors that will be ever used:
input_mesh_color = grey_1
curve_color = orange
our_color = orange
repulsive_curves_color = dark
point_cloud_default_color = pink
vector_field_default_color = grey_2
default_edge_color = grey_2