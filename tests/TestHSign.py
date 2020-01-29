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

try:
    print(x[11])
except IndexError as err:
    print('Caught exception: ' + str(err))
