# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:37:46 2022

@author: max90
"""


import cv2
import numpy as np
from PIL import Image
import os


def al_grid():
    """
    Converted a grid into images of the grid
    """
    img = cv2.imread("al_grid.png")  # 88
    os.chdir("letter_grid")
    inc = 255
    sv = 89
    # img= cv2.line(img,(sv+inc,0), (sv+inc,1000),(0,0,255),thickness=1) #numpy it is row, col, but the cv write is col,row
    sh = 88
    inch = 306
    # img= cv2.line(img,(0,sh+inch), (1000,sh+inch),(255,0,255),thickness=1)
    for col in range(1, 8):
        s = "!VWXYZ     "
        l_img = img[
            sh + inch * 3 : sh + inch * 4, sv + (col - 1) * inc : sv + inc * col + 4
        ]
        # cv2.imwrite(f"{s[col]}.jpg", l_img)
        img_p = Image.fromarray(l_img, "RGB")  # ratio is 33:28  row:col
        img_p.show()


def load_tiles():
    """
    returns dict of alpha:img of tile
    """
    os.chdir("letter_grid")
    al_img = dict()
    for file in os.listdir():
        np_arr = cv2.imread(file)
        # img_p = Image.fromarray(np_arr, 'RGB') #ratio is 33:28  row:col
        # img_p.show()
        al_img[(file[0]).lower()] = np_arr
    return al_img


# al_img = load_tile


# al_grid
# img = Image.fromarray(img, 'RGB')
# img.show()
