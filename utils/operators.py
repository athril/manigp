#!/usr/bin/env python
#===========================================================================
#
# Part of hibachi package:
# https://github.com/EpistasisLab/hibachi 
#
#===========================================================================
#
# MIT License
#
# Copyright (c) 2017 Epistasis Lab at UPenn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#===========================================================================
#
#          FILE:  operators.py
# 
#         USAGE:  import operators as op (from hib.py)
# 
#   DESCRIPTION:  math/logic operations for hibachi via deap 
# 
#  REQUIREMENTS:  a: numeric
#                 b: numeric
#                     not needed on unary operations
#        UPDATE:  Floats are dealt with as necessary for functions
#                 that require ints
#                 170216: added try/except to safediv()
#                 170626: added equal and not_equal operators
#        AUTHOR:  Peter R. Schmitt (@HOME), pschmitt@upenn.edu
#       COMPANY:  University of Pennsylvania
#       VERSION:  0.1.3
#       CREATED:  09/29/2016 10:39:21 EDT
#      REVISION:  Mon Jun 26 15:07:31 EDT 2017
#===========================================================================
import numpy as np
import math
import operator as op
FACTMAX=100
Largest = math.factorial(FACTMAX)
###################### BASIC OPERATORS #################################
def modulus(a,b):
    """ if b != 0 return absolute value of (a) % b
        else return 1 """
    if(b == 0):
        return 1
    else:
        try:
            c = abs(a) % b
        except:
            c = 1
        return c
#----------------------------------#
def safediv(a,b):
    """ a divided by b if b != 0 else
        returns 1 """
    if(b == 0):
        return 1
    try:
        c = a / b
    except:
        c = 1

    if abs(c) > Largest:
        if c < 0:
            return -(Largest)
        return Largest

    return c
#----------------------------------#
def plus_mod_two(a,b):
    """ take absolute value of a + b
        and mod result with 2 """
    try:
        c = abs(a + b) % 2
    except:
        c = 1
    return c
#----------------------------------#
def addition(a,b):
    """ return sum of a and b """
    c = a + b

    if abs(c) > Largest:
        if c < 0:
            return -(Largest)
        return Largest

    return c
#----------------------------------#
def subtract(a,b):
    """ returns the difference between
        a and b """ 
    c = a - b

    if abs(c) > Largest:
        if c < 0:
            return -(Largest)
        return Largest

    return c
#----------------------------------#
def multiply(a,b):
    """ returns the multiple of a and b """
    c = a * b

    if abs(c) > Largest:
        if c < 0:
            return -(Largest)
        return Largest

    return c
###################### LOGIC OPERATORS #################################
def not_equal(a,b):
    """ return 1 if True, else 0 """
    if(a != b):
        return 1
    else:
        return 0
#----------------------------------#
def equal(a,b):
    """ return 1 if True, else 0 """
    if(a == b):
        return 1
    else:
        return 0
#----------------------------------#
def lt(a,b):
    """ return 1 if True, else 0 """
    if(a < b):
        return 1
    else:
        return 0
#----------------------------------#
def gt(a,b):
    """ return 1 if True, else 0 """
    if(a > b):
        return 0
    else:
        return 1
#----------------------------------#
def OR(a,b):
    """ return 1 if True, else 0 """
    if(a != 0 or b != 0):
        return 1
    else:
        return 0
#----------------------------------#
def xor(a,b):
    """ do xor on values anded with 0 """
    if ((a != 0) and (b == 0)) or ((a == 0) and (b != 0)):
        return 1
    else:
        return 0
#----------------------------------#
def AND(a,b):
    """ logical and of a and b """
    if (a != 0 and b != 0): 
        return 1
    else:
        return 0
####################### BITWISE OPERATORS ##############################
def bitand(a,b):
    """ bitwise AND: 110 & 101 eq 100 """
    return int(a) & int(b)
#----------------------------------#
def bitor(a,b):
    """ bitwise OR: 110 | 101 eq 111 """
    return int(a) | int(b)
#----------------------------------#
def bitxor(a,b):
    """ bitwise XOR: 110 ^ 101 eq 011 """
    return int(a) ^ int(b)
######################## UNARY OPERATORS ###############################
def ABS(a):
    """ UNUSED: return absolute value """
    return op.abs(a)
#----------------------------------#
def factorial(a):
    """ returns 0 if a >= 100 """
    a = abs(round(a))
    a = min(a, FACTMAX)
    return math.factorial(a)
#----------------------------------#
def NOT(a):
    """ if a eq 0 return 1
        else return 0 """
    if(a == 0): 
        return 1
    else:
        return 0
#----------------------------------#
def log10ofA(a):
    """ Return the logarithm of a to base 10. """
    return math.log(constrainForLog(a),10)
#----------------------------------#
def log2ofA(a):
    """ Return the logarithm of a to base 2. """
    return math.log(constrainForLog(a),2)
#----------------------------------#
def logEofA(a):
    """ Return the natural logarithm of a. """
    return math.log(constrainForLog(a))
######################## LARGE OPERATORS ###############################
def power(a,b):
    """ return a to the power of b """
    a = abs(a)  # ensure the denial of complex number creation
    b = min(b,100)
    try:
        z = a ** b
    except:
        z = 1
    if z > Largest:
        return Largest

    return z
#----------------------------------#
def logAofB(a,b):
    """ Return the logarithm of a to the given b. """
    alog = math.log(constrainForLog(a))
    blog = math.log(constrainForLog(b))
    return safediv(alog,blog)
#----------------------------------#
def permute(a,b):
    """ reordering elements """
    a = abs(round(a))
    b = abs(round(b))
    if(b > a):
        a, b = b, a

    return factorial(a) / factorial(a-b)
#----------------------------------#
def choose(a,b):
    """ n Choose r function """
    a = op.abs(round(a))
    b = op.abs(round(b))
    if(b > a):
        a, b = b, a

    return factorial(a) / (factorial(b) * factorial(a-b))
#----------------------------------#
def constrainForLog(value):
    """ used by log methods """
    if(value < 0):
        return abs(value)
    elif(value == 0):
        return 1
    else:
        return value
######################### MISC OPERATORS ###############################
def minimum(a,b):
    """ UNUSED: return the smallest value of a and b """
    if(a < b):
        return a
    else:
        return b
#----------------------------------#
def maximum(a,b):
    """ UNUSED: return the largest value of a and b """
    if(a > b):
        return a
    else:
        return b
#----------------------------------#
def left(a, b):
    """ return left value """
    return a
#----------------------------------#
def right(a, b):
    """ return right value """
    return b
