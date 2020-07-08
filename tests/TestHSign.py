# Copyright 2020 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# TestHSign.py
# Written Ian Rankin - January 2020
#
#



import rdml_graph as gr




# Create HSignature

x = gr.HSignature(10)
y = gr.HSignature(10)
z = gr.HSignature(10)

print('x[5:7] = ' + str(x[5:7]))

print('len(x) = ' + str(len(x)))
print('x == y = ' + str(x == y))
print('x != y = ' + str(x != y))
print('x[5] = ' + str(x[5]))

x.cross(5, 1)
print('x[5] after cross = ' + str(x[5]))
print('x == y = ' + str(x == y))
print('x != y = ' + str(x != y))
print('x[5] = ' + str(x[5]))

print('x == z = ' + str(x == z))

x.cross(4,-2)
print(x)
x.cross(5,-1)
print(x)
x.cross(5, -1)
print(x)
x.cross(5, -1)
print(x)


print('x = ' + str(x))
t = x.copy()
print('t = ' + str(t))
print('CHANGE')
t.cross(3,-1)
print('x = ' + str(x))
print('t = ' + str(t))

try:
    print(x[11])
except IndexError as err:
    print('Caught exception: ' + str(err))
