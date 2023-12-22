# -*- coding: utf-8 -*-
"""
Created on Wed May 25 18:40:27 2022

@author: max90
"""
import pickle

with open("scrab.txt","r") as f:
    l= f.read().split('\n')
    g = []
    for word in l:
        word = word.lower()
        valid=False
        for i in 'aeiouy':
            if i in word: #no duplicates
                valid=True
                break
        if "'" in word:
            valid=False
        
        if valid:
            g.append(word)

word_dict = dict()
for word in g:
    srt=''.join(sorted(word)) #alphabitizes
    if srt not in word_dict:
        word_dict[srt] = [word]
    else:
        word_dict[srt].append(word)


def save():

    with open('saved_dictionary.pkl', 'wb') as f:
        pickle.dump(word_dict, f)
        
    with open('saved_dictionary.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)








alpha="abcdefghijklmnopqrstuvwxyz"
alpha_dict = {i:0 for i in alpha}
from copy import deepcopy


def save2():

    with open('saved_length.pkl', 'wb') as f:
        pickle.dump(length_dict, f)
        
    with open('saved_length.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)

length_dict = dict()
for alphabitized in word_dict:
    c_dict={i:0 for i in alphabitized}
    for char in alphabitized:
        c_dict[char]+=1
    if len(alphabitized) not in length_dict:       #c_dict holds only anagram. You will need to use word_dict to get the words from that anagram
        length_dict[len(alphabitized)] = [c_dict] #anagram_length: dict(char:count)
    else:
        length_dict[len(alphabitized)].append(c_dict) 
        #so I can't alphabitize and check in as there could be extranenous letters. 
        #So you take your unused, and count. Then for each word, longer words first, you check if the unused has all the letters as a word. 
        #1: take your unused string and use an alphadict to that has every letter initially set to zero, and for each character +=1 to value.
        #2: go throughy the length_dict by key (letter) and test if alphadict has equal to or greater for each letter in the alphabtized. 
        #So O time is num words * the letters (no duplicates) 
        #the letters no dup worst case scenerio for all of them lets call it 8. on average, it is. 
        #I feel like any time you are using combinations it will be slower or equal. 
        #How could combos be faster?  15 C 14 is 15 steps max, but you could have more then 15 words anagrams
        # so 4 C 3 is 4, but you will have 201 words. So this is a case where the combos (4<201) is way better. 
        #
    
        
def count_non_repeat() :
    a=0
    for i in word_dict:
        tmp=""
        for char in i:
            if char not in tmp:
                tmp+=char
                a+=1
    print(a/len(word_dict))
    
save()
save2()
        
        
