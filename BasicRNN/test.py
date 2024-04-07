import numpy as np
import rnn_utils

def print_type_and_value(x):
  print(type(x))
  print(x)

a = np.array([1,2,3,4])
print(a, type(a))
# [1 2 3 4] <class 'numpy.ndarray'>
b = rnn_utils.sigmoid(a)
print(b, type(b))
# [0.73105858 0.88079708 0.95257413 0.98201379] <class 'numpy.ndarray'>
c = rnn_utils.softmax(a)
print(c, type(c))
# [0.0320586  0.08714432 0.23688282 0.64391426] <class 'numpy.ndarray'>

a = np.array([[1,2],[3,4]])
print(a, type(a))
# [[1 2]
#  [3 4]] <class 'numpy.ndarray'>
b = rnn_utils.sigmoid(a)
print(b, type(b))
# [[0.73105858 0.88079708]
#  [0.95257413 0.98201379]] <class 'numpy.ndarray'>
c = rnn_utils.softmax(a)
print(c, type(c))
# [[0.11920292 0.11920292]
#  [0.88079708 0.88079708]] <class 'numpy.ndarray'>
print_type_and_value(c[0,:])
# <class 'numpy.ndarray'>
# [0.11920292 0.11920292]
print_type_and_value(c[:,0])
# <class 'numpy.ndarray'>
# [0.11920292 0.88079708]

print("# softmax")
x = np.array([[1,2],[3,5]])
e_x = np.exp(x - np.max(x))
r = e_x / e_x.sum(axis=0)
print_type_and_value(e_x)
print_type_and_value(e_x.sum(axis=0))
print_type_and_value(r)

print("# divide")
print_type_and_value(np.array([1,2])/np.array([3,4]))
# <class 'numpy.ndarray'>
# [0.33333333 0.5       ]
print_type_and_value(np.array([[1,2],[3,4]])/np.array([3,4]))
# <class 'numpy.ndarray'>
# [[0.33333333 0.5       ]
#  [1.         1.        ]]