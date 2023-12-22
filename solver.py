# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:50:50 2022

@author: max90
"""
from copy import deepcopy
import pickle
from timeit import default_timer as timer
from itertools import combinations
import math
import cv2
import numpy as np
from PIL import Image
import os
import bisect

with open("saved_dictionary.pkl", "rb") as f:
    word_dict = pickle.load(f)  # alphabitized anagram:[words]

with open("saved_length.pkl", "rb") as f:
    length_dict = pickle.load(f)

alpha = "abcdefghijklmnopqrstuvwxyz"
MAX_LENGTH = 6  # max(length_dict)
num_anagram = len(word_dict)
best_st = None  # need to adding a letter
best_num_unused = 100000
size_grid = 64
wait_time = 5  # The amount of time to wait from start_time to timer() in solve function


# The method takes words and converts them into an alphabitized anagram. Them when you have another anagram, you convert it into alphabetical order and see if there is a word with the same anagram
# genius


# 4 This configuration lead to a dead end. Now you want to keep as much of the structure intact, so there a two ways. First, keep as much of the structure intact as possible.
# Second, you could choose to keep the longer words or you could keep the words that have the most positions for other words.


# You might actually be able to fit stuff inbetween, but I am lazy
class w:
    num = [0]

    def __init__(self, word, coords, orin, con_word=None):
        # maybe can just store a global of how many words have ever existed, so that way ID isn't repeated.
        self.word = word
        self.coords = coords  # should be a list of coordnites that this word occupies. Indexes need to be aligned with self.word. Up to down, left to right
        self.coords_adj = set()  # use set to generate all the coords and adjacent
        for c in coords:
            for ac in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                self.coords_adj.add((c[0] + ac[0], c[1] + ac[1]))

        self.orin = orin  # h for horizontal(row), v for vertical(column).
        self.id = self.num[0]
        self.num[
            0
        ] += 1  # this will allow you to have duplicate words and distinguishduplicate words

        # So each time a new word is connected to the base word, that will count as a new branch. I'm not sure how to structure the branches. Probably just start with one word, the base word, and then attached in a list will be all branches the words that connect to the base word.
        if con_word == None:
            self.branches = (
                []
            )  # list of words that connect to this word. A new word has no branches as nothing has yet to attach to it.
        else:
            self.branches = [
                con_word
            ]  # you will keep the biggest branch first, and all the smaller ones get delated.

    def add(self, t):
        self.branches.append(t)

    def get_tiles_branch(self, parent_branch_id=-1):
        """
        This is absolute garbage as it does not take into account repeating letters

        Returns:
            returns this words tiles and all the tiles of the branches that depend on it.
            So how many unused letters if this word and sub branches were delated
        """
        s = ""
        s += self.word
        for subword in self.branches:
            if subword.id != parent_branch_id:
                s += subword.get_tiles_branch(self.id)
        return s

    def get_words_and_used(self, parent_branch_id=-1, l_words=None, used=""):
        """
        Gets words and all the used tiles. Doesn't include self.word in used. It also doesnt include the connecting letters of self.word
        """
        if l_words == None:
            l_words = []
        for subword in self.branches:
            if subword.id != parent_branch_id:
                if self.orin == "h":
                    con_coord = (self.coords[0][0], subword.coords[0][1])
                else:
                    con_coord = (subword.coords[0][0], self.coords[0][1])
                ind = self.coords.index(con_coord)
                con_l = self.word[ind]
                used += subword.word.replace(con_l, "", 1)
                l_words.append(subword)
                n_words, n_used = subword.get_words_and_used(self.id)
                used += n_used
                l_words += n_words
        return l_words, used


class struct:
    def __init__(self, word):
        """
        Initializes a struct based off a first word that is off class w
        """
        self.words = [word]
        self.cl = set(
            word.word
        )  # all the connecting letters. no dups. Helps with effiency

        self.coords = []
        self.coords += word.coords

        self.coords_adj = []
        self.coords_adj += word.coords_adj

        self.used_tiles = word.word

    def add(self, word, added_tiles):
        """
        word is of class w
        added_tiles are tiles that were unused but are now on the struc
        Function updates the structure by adding word
        """
        self.words.append(word)
        self.used_tiles += added_tiles
        self.coords += word.coords
        self.coords_adj += word.coords_adj
        for char in word.word:
            self.cl.add(char)

    def get_unused(self, tiles):
        unused = tiles
        for char in self.used_tiles:
            unused = unused.replace(char, "", 1)
        return unused


def gen_words(num_to_use, unused, cl=set(alpha)):
    """
    This will add connecting letters. Can use cl=[""] to just get words
    num_to_use: how many unused letters to use. The words will be length of num_to_use+1
    cl: a set of the connecting letters

    Otime(len(cl)*num_comb or len(cl)*num_anagram)

    Returns:
        connecting_map, which is a dict that uses each letter as a key, and the value are all words of length num_to_use+1 that can be made
        if you combine the connecting letter and the unused
    """
    # instead of alpha, I should just look at the connecting letters
    s = timer()
    inc = 1
    if cl == [""]:
        inc = 0

    num_anagrams = len(length_dict[num_to_use + inc])
    num_comb = math.comb(len(unused), num_to_use)
    con_map = {f"{i}": [] for i in cl}

    # print(num_comb,num_anagrams," comb,anagram")
    # num_anagrams=1

    if num_comb < num_anagrams:
        for al in cl:
            for com in combinations(
                unused, num_to_use
            ):  # problem is that the same combo will come up, so you will have duplicate words in con_map
                anagram = "".join(
                    sorted("".join(com) + al)
                )  # the words themselves are num_to_use+1, as num_to_use just represents number of unused to use
                if anagram in word_dict:
                    con_map[al] += word_dict[anagram]
    else:
        unused_count = {
            i: 0 for i in alpha
        }  # Use full alpha dict for this, so you can't get a lookup error
        unused_count[""] = 0
        for char in unused:
            unused_count[
                char
            ] += 1  # This just counts how many times each letter comes up
        for al in cl:
            unused_count[
                al
            ] += 1  # Could change length_dict: (dict , where has_char:c_dict)
            for c_dict in length_dict[
                num_to_use + inc
            ]:  # c_dict is a count dict that represents a valid anagram. #here I loop through words
                # could redo length_dict so I don't have to iterate through unnescarry
                if al in c_dict or al == "":  # it has that connecting letter
                    valid = True
                    for (
                        char
                    ) in (
                        c_dict
                    ):  # make sure that you have at least the tiles to make the word
                        if unused_count[char] < c_dict[char]:
                            valid = False
                            break
                    if valid:  # there are enough tiles for this word
                        anagram = ""
                        for char in c_dict:
                            for dup in range(c_dict[char]):
                                anagram += char  # c_dict should already be ordered
                        con_map[al] += word_dict[
                            anagram
                        ]  # stores words that have al in it, can be made with the unused + al, and are of num_to_use st+1
            unused_count[al] -= 1
    e = timer()
    # print(e-s,"gen_words")
    return con_map


def solve(st, tiles, start_timer=1e99):
    global cur_st, best_st, best_num_unused  # cur_s is the structure in real life
    if timer() - wait_time > start_timer:
        if len(st.get_unused(tiles)) < best_num_unused:
            best_num_unused = len(st.get_unused(tiles))
            best_st = st
        return False, best_st

    unused = st.get_unused(tiles)
    cl = st.cl

    if len(unused) == 0:
        best_num_unused = 0
        best_st = st
        return True, st

    unused_count = {
        i: 0 for i in alpha
    }  # Use full alpha dict for this, so you can't get a lookup error
    for char in unused:
        unused_count[char] += 1  # This just counts how many times each letter comes up

    if len(unused) >= MAX_LENGTH:  # MAX_LENGTH is the largest key in length_dict
        start = (
            MAX_LENGTH - 1
        )  # st represents the amount of unused letters you could use. If you have 14 unused, then you can only use 13 as there is one for connecting and the largest word is len==14
    else:
        start = len(unused)
    s = timer()
    itera = sorted(length_dict.keys())
    itera.reverse()
    for length in range(
        start, 0, -1
    ):  # length represents the amount of unused tiles that will be used
        con_map = gen_words(
            length, unused, cl
        )  # So if we have con_letter, use con_map[con_letter] to see what words we could fit
        # Now after getting the words for each length, check positions
        # Loop through each word in the struct.Then for con letter in the word, use con_map to place words. s.count(con_letter) could be >1, so try each posttion
        for word in st.words:  # loop through each word in structure
            # print(word.word)
            invalid_coords = st.coords_adj.copy()  # tuples are immutable
            for coord in word.coords_adj:
                invalid_coords.remove(
                    coord
                )  # invalid_coords contains all the coords from all words in st.words except the coords of word

            # print(invalid_coords)
            # print(con_map)
            for ind in range(len(word.word)):  # loop through the connecting letter
                con_l = word.word[ind]
                con_coord = word.coords[ind]
                if len(con_map[con_l]) != 0:
                    for possible_word in con_map[con_l]:
                        con_l_index = -1
                        for repos in range(possible_word.count(con_l)):
                            con_l_index = possible_word.find(
                                con_l, con_l_index + 1
                            )  # this index is for a possible word, and not the base word
                            tiles_to_dr = length - con_l_index
                            tiles_to_ul = con_l_index

                            new_coords = []
                            if word.orin == "h":  # attaching word is "v"
                                orin_new = "v"
                                for i in range(-tiles_to_ul, tiles_to_dr + 1):  #
                                    new_coords.append((con_coord[0] + i, con_coord[1]))
                            else:
                                orin_new = "h"
                                for i in range(-tiles_to_ul, tiles_to_dr + 1):  #
                                    new_coords.append((con_coord[0], con_coord[1] + i))
                            valid = True
                            for i in new_coords:
                                if i in invalid_coords:
                                    valid = False
                                    break
                            if valid:
                                new_st = deepcopy(st)
                                # find the word in new_st
                                for loc_word in new_st.words:
                                    if (
                                        loc_word.id == word.id
                                    ):  # find the old word in new_st
                                        new_word = w(
                                            possible_word,
                                            new_coords,
                                            orin_new,
                                            loc_word,
                                        )
                                        loc_word.add(new_word)
                                        new_st.add(
                                            new_word,
                                            new_word.word.replace(con_l, "", 1),
                                        )
                                        ans, st_solved = solve(
                                            new_st, tiles, start_timer
                                        )
                                        if ans:
                                            best_num_unused = 0
                                            best_st = st_solved
                                            return True, st_solved

    if len(unused) < best_num_unused:
        best_num_unused = len(unused)
        best_st = st
    return False, st


def load_tiles(size):
    """
    returns dict of alpha:img of tile
    """
    # os.chdir("letter_grid")
    al_img = dict()
    directory = "letter_grid"
    for file in os.listdir(directory):
        np_arr = cv2.imread(os.path.join(directory, file))
        if file == "unused.png":
            np_arr = cv2.resize(np_arr, (size * 3, size))
            al_img["unused"] = np_arr
        else:
            np_arr = cv2.resize(np_arr, (size, size))
            al_img[(file[0]).lower()] = np_arr
        # img_p = Image.fromarray(np_arr, 'RGB') #ratio is 33:28  row:col
        # img_p.show()

    return al_img


def show_st(st, prev_st=None):
    """
    Takes st, and displays img of board
    """
    l_most = 0
    u_most = 0
    r_most = 0
    d_most = 0
    for placed_tile in st.coords:
        if placed_tile[0] < u_most:
            u_most = placed_tile[0]

        if placed_tile[1] < l_most:
            l_most = placed_tile[1]

        if placed_tile[1] > r_most:
            r_most = placed_tile[1]

        if placed_tile[0] > d_most:
            d_most = placed_tile[0]

    # print(u_most,l_most,d_most,r_most)
    width = r_most - l_most + 1  # This is really the amount of cols
    length = d_most - u_most + 1  # amount of rows

    al_img = load_tiles(size_grid)
    unused = st.get_unused(tiles)
    if width < len(unused) + 3:
        width = len(unused) + 3  # unused: takes up 3 tiles

    img = np.zeros(
        [length * size_grid + 2 * size_grid, width * size_grid, 3], dtype=np.uint8
    )
    img += 225

    coord_letter = dict()  # coordnite to letter
    unnorm_coord_letter = dict()  # coordnite to letter
    for word in st.words:
        for i in range(len(word.word)):
            c = word.coords[i]
            unnorm_coord_letter[(c[0], c[1])] = word.word[i]
            norm_coord = (c[0] - u_most, c[1] - l_most)
            coord_letter[norm_coord] = word.word[i]

    if prev_st == None:
        for coord in coord_letter:
            row = coord[0]
            col = coord[1]
            img[
                row * size_grid : (row + 1) * size_grid,
                col * size_grid : (col + 1) * size_grid,
            ] = al_img[coord_letter[coord]]
    else:
        same = []
        for word in prev_st.words:
            for i in range(len(word.word)):
                c = word.coords[i]
                if c in unnorm_coord_letter:
                    if unnorm_coord_letter[c] == word.word[i]:
                        same.append(c)
        for coord in unnorm_coord_letter:
            norm_coord = (coord[0] - u_most, coord[1] - l_most)
            row = norm_coord[0]
            col = norm_coord[1]
            if coord in same:
                img[
                    row * size_grid : (row + 1) * size_grid,
                    col * size_grid : (col + 1) * size_grid,
                ] = (
                    al_img[coord_letter[norm_coord]] * 0.4
                )
            else:
                img[
                    row * size_grid : (row + 1) * size_grid,
                    col * size_grid : (col + 1) * size_grid,
                ] = al_img[coord_letter[norm_coord]]

            # norm_coord = (c[0]-u_most,c[1]-l_most)
            # prev_unnorm_coord_letter[norm_coord]=word.word[i]

    brow = img.shape[0]
    img[(brow - size_grid) : brow, 0 : 3 * size_grid] = al_img["unused"]
    for u in range(len(unused)):
        img[
            (brow - size_grid) : brow, (3 + u) * size_grid : (4 + u) * size_grid
        ] = al_img[unused[u]]

    img_p = Image.fromarray(img, "RGB")  # ratio is 33:28  row:col
    img_p.show()
    return img_p


def start(tiles):
    """
    Starts from no words and tries to find a working structure. Will use solve
    """
    global best_st, best_num_unused
    if len(tiles) > MAX_LENGTH:
        start = MAX_LENGTH
    else:
        start = len(tiles)
    start_timer = timer() + wait_time / 2
    for length in range(start, 1, -1):
        con_map = gen_words(length, tiles, [""])
        # print(con_map,length)
        for word_string in con_map[""]:
            coords = [(0, i) for i in range(length)]  # horizontal,row
            new_w = w(word_string, coords, "h")
            new_st = struct(new_w)
            ans, solve_st = solve(new_st, tiles, start_timer)
            if ans:
                return True, solve_st
            elif len(solve_st.get_unused(tiles)) < best_num_unused:
                best_num_unused = len(solve_st.get_unused(tiles))
                best_st = solve_st
    return False, best_st


def gen_st(word):
    """
    takes word of class word and returns an st from that word
    """

    l_words, used = word.get_words_and_used(l_words=[word])

    new_st = struct(l_words[0])
    for word in l_words[1:]:
        new_st.add(word, "")

    # print(new_st.used_tiles,'\n')
    new_st.used_tiles += used

    return new_st


def solve_remove(inp_st):  # I forgot to optimize based on
    global best_st, best_num_unused

    # So you start by keeping the most/removing the least. So for the input_st you would remove 1 word. Then choose the best with keeping the most, called a
    # Try to solve. If that doesn't work, update the search query by removing 1 word from a.

    def update_query(st):
        for word in st.words:
            for sw in word.branches:
                sub_word = deepcopy(sw)
                for i in range(
                    len(sub_word.branches)
                ):  # So this removes the parent branch, allowing menance activities
                    if sub_word.branches[i].id == word.id:
                        sub_word.branches.pop(
                            i
                        )  # removes the connecting word, and the remaining structure is left
                        break
                new_st = gen_st(
                    sub_word
                )  # So will catalogous be len(new_st.used_tiles).
                # So new_st becomes a word that was kept.
                ind = bisect.bisect(query_num, len(new_st.used_tiles))
                query_num.insert(ind, len(new_st.used_tiles))
                query_st.insert(ind, new_st)
        return query_num, query_st

    query_st = []
    query_num = (
        []
    )  # goes from least number of used_letters , to most number of used_letters. You want to keep the most letters
    update_query(inp_st)  # mutable data types
    while (
        len(query_num) != 0 and query_num[-1] / (len(tiles) - best_num_unused)
    ) > 0.5:
        query_num.pop(-1)
        tmp_st = query_st.pop(-1)
        ans, s_st = solve(tmp_st, tiles, timer() + wait_time / 2)
        if ans:
            best_st = s_st
            best_num_unused = 0
            return True, s_st
        if len(s_st.get_unused(tiles)) < best_num_unused:
            best_st = s_st
            best_num_unused = s_st.get_unused(tiles)
        update_query(s_st)

    return False, best_st
    # Search
    # If not, update the queries by removing a word. Basically copy code from above

    # So then, you examine the best one, which will be found using .pop(-1). Then you try to solve.
    # If solution found, return. Else, check if this is the best structure. Also, make it so that if two are equal, it keeps the first
    # Or maybe if a structure is not found without keepings lets say 50% of .used_tiles to tiles, then it gives you the option
    # to keep going or you can dump. Honestly, it is prob best to dump if you removed 50% of letters and can still not find a solution
    # the dump will be quicker then reorginzing the entire strucutre. Maybe it has options, like tried 75, continue? tried 50, continue?


# So you have your 21 starting letters. You then get the structure for that.
# Assuming you get split, now you call solve again with what was just the prev_solved st and then with the new letter added to tiles
# If you can't add the new letter to the structure, you then go through structure and remove words.
# You can only keep one branch. So for each word, then in inner loop keep a branch
# Keep the biggest branches first. Mannnnn, what if you want to delate two supporting branches. Maybe I can have an ordered list, of options, that way it searches the way to keep the struct first. I guess it will eventually delate supporting word. Like it will keep taking words off from struct.

# Tiles are constant.
# Given unsolved st, loop through st.words  for word in st.words
# Loop through branch in the word         for word_branch in word.branch  #each element in word.branch is a word
# Now in that branch, I have to delate the connecting word  word_branch.branch.delate(word)
# Now that the parent word has been delated from subword, I need a way to generate the new_st
# I should be able to use st(word_branch) as my first word. Then recursively add the words in .branch to st. This should work.
# I will test by having a struct, and seeing if I can generate a new struct after delating a word with only one branch


# I guess loop through each word in struct, and then loop through branch. I need a way to go from a branch to a struct
# solve can't go backwards, only foward, so I need a way to go backwards


if __name__ == "__main__":
    tiles = input("enter the starting tiles:")
    saved_tiles = tiles
    cur_st = None
    ans, best_st = start(tiles)
    show_st(best_st)
    saved_st = best_st
    while True:
        sinp = input(
            "\na to add tiles\nr to redo to last state\nd if dump and remove from unused \nz delete letters and restart\n anything else show tiles:"
        )
        if "a" == sinp:
            inp = input("enter the letters adding:")
            best_num_unused += len(inp)
            saved_tiles = tiles
            tiles += inp
        elif sinp == "d":  # Signifies a dump. Only use this if the letter is unused
            inp = input(
                "enter one letter to delate from unused:"
            )  # Should only ever delate an unused tile
            best_num_unused -= 1
            saved_tiles = tiles
            tiles = tiles.replace(inp, "", 1)

        elif sinp == "r":
            best_num_unused = len(saved_st.get_unused(tiles))
            tiles = saved_tiles
            best_st = saved_st
            print(tiles)
            show_st(best_st)
            continue
        elif sinp == "z":
            # restarts everything. Don't make a typo!
            inp = input("enter letters to delate:")
            best_st = saved_st
            best_num_unused = len(saved_st.get_unused(tiles))
            saved_tiles = tiles
            for char in inp:
                tiles = tiles.replace(char, "", 1)
            ans, best_st = start(tiles)
            print(tiles)
            show_st(best_st)
            continue

        else:
            print("tiles: " + "".join(sorted(tiles)))
            continue

        # Try to solve again with the updated tiles
        saved_st = deepcopy(best_st)

        ans, best_st = solve(best_st, tiles, timer())

        if not ans:
            ans, best_st = solve_remove(best_st)
            if (
                not ans
            ):  # Failed when trying to remove words. Need to give each removal of words some time.
                inp = input(
                    "Fail to easily update structure. s to find solution, else continue:"
                )
                if inp == "s":
                    best_st = None
                    best_num_unused = 1e99
                    ans, best_st = start(tiles)

        show_st(best_st, saved_st)


# problems:
#    placing a word that could connect in between words. This violates only one adjacent
# when fitting word inbetween, you can check the endpoints and see if that is a word

# should I priotize words  that are rarer, that use more unique letters? I guess it won't matter as eventually that option will be in the search algorithim

# wastes time on checking coords that def don't exist, maybe tries placing on words that only have one connecting branch. This way it is likely to give you more valid connecting letters to find a strucutre.
# doesn't try to create the largest structure possible.

# Give maybe 10 seconds to solve, and then after that, abort the program and just use best_st and dump the unused letter. Maybe have a global variable for current time and check in solve if 10 seconds have passed

# the only thing left is to color code the changes

# The longest a word can be is 6 letters, so chunk three letter chunks
# I think I should make it so that I can see what the prev structure was, so that I can see what was kept and what was not
