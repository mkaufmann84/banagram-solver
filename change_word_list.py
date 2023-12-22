# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:35:21 2022

@author: max90
"""
alpha="abcdefghijklmnopqrstuvwxyz"

def check_word(word):
    for char in word:
        if char not in alpha:
            return False
    return True

with open("words.txt",'r') as f:
    l= f.read().split('\n')
    g=[]
    for word in l:
        if check_word(word):
            g.append(word)
        else:
            print(word)
with open("words.txt","w") as f:
    gg = []
    for i in g:
        if i not in gg:
            gg.append(i)
    f.write("\n".join(gg))
    


