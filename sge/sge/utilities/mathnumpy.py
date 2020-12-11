from math import log,exp,sqrt,sin,cos,tan, cosh, sinh
import numpy as np

distance = np.zeros((257,257))
for i in range(257):
  for j in range(257):
    distance[i,j] = np.sqrt(pow(i - 128, 2) + pow(j - 128, 2)) / np.sqrt(pow(128, 2) + pow(128, 2))

def _log_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      if x[i] <= 0: 
        x[i] = 0
      else:
        x[i] = log(x[i])
    return x
  else:
    if x <= 0: 
      return 0
    return log(x)
    
def _sin_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      try:
        x[i] = sin(x[i])
      except:
        x[i] = 0
    return x
  else:
    try:
      return sin(x)
    except:
      return 0


def _cos_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      try:
        x[i] = cos(x[i])
      except:
        x[i] = 0
    return x
  else:
    try:
      return cos(x)
    except:
      return 0
    
    
def _sinh_(x):
  if isinstance(x, list):
    for i in range(len(x)):
        try:
          x[i] = sinh(x[i])
        except:
          x[i] = 1.0
    return x
  else:
    if x <= 0: 
      return 0
    try:
      return sinh(x)
    except:
      return 1

def _cosh_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      try:
        x[i] = cosh(x[i])
      except:
        x[i] = 1.0 
    return x
  else:
    try:
      return cosh(x)
    except:
      return 1.0


def _tan_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      if x[i] <= 0: 
        x[i] = 0
      else:
        x[i] = tan(x[i])
    return x
  else:
    if x <= 0: 
      return 0
    return tan(x)


def protdiv(x,y):
  if isinstance(x, list):
    if isinstance(y, list):
      for i in range(min(len(x), len(y))):
        if y[i] == 0: 
          continue
        x[i] /= y[i]
      return x
    else:
      for i in range(len(x)):
          if y == 0: 
            continue
          else:
            x[i] /= y
      return x
  elif isinstance(y, list):
    for i in range(len(y)):
        if y[i] == 0: 
          continue
        else:
          y[i] = x / y[i]
    return y
  else:
    if(y==0): 
      return x
    return x / y
    
def ssum(x,y):
  if isinstance(x, list):
    if isinstance(y, list):
      for i in range(min(len(x), len(y))):
        try:
          x[i] = int(x[i]) + int(y[i])
        except:
          x[i] = 0
        
      return x
    else:
      for i in range(len(x)):
        try:
          x[i] = int(x[i]) + int(y)
        except:
          x[i] = 0
      return x
  elif isinstance(y, list):
    for i in range(len(y)):
      try:
        y[i] = int(x) + int(y[i])
      except:
        y[i] = 0
    return y
  else:
    try:
      return int(x) + int(y)
    except:
      return 0

def ssub(x,y):
  if isinstance(x, list):
    if isinstance(y, list):
      for i in range(min(len(x), len(y))):
        try:
          x[i] = int(x[i]) - int(y[i])
        except:
          x[i] = 0
        
      return x
    else:
      for i in range(len(x)):
        try:
          x[i] = int(x[i]) - int(y)
        except:
          x[i] = 0
      return x
  elif isinstance(y, list):
    for i in range(len(y)):
      try:
        y[i] = int(x) - int(y[i])
      except:
        y[i] = 0
    return y
  else:
    try:
      return int(x) - int(y)
    except:
      return 0


def smul(x,y):
  if isinstance(x, list):
    if isinstance(y, list):
      for i in range(min(len(x), len(y))):
        try:
          x[i] = int(x[i]) * int(y[i])
        except:
          x[i] = 1
        
      return x
    else:
      for i in range(len(x)):
        try:
          x[i] = int(x[i]) * int(y)
        except:
          x[i] = 1
      return x
  elif isinstance(y, list):
    for i in range(len(y)):
      try:
        y[i] = int(x) * int(y[i])
      except:
        y[i] = 1
    return y
  else:
    try:
      return int(x) * int(y)
    except:
      return 1
    
def sxor(x,y):
  if isinstance(x, list):
    if isinstance(y, list):
      for i in range(min(len(x), len(y))):
        try:
          x[i] = int(x[i]) ^ int(y[i])
        except:
          x[i] = 0
        
      return x
    else:
      for i in range(len(x)):
        try:
          x[i] = int(x[i]) ^ int(y)
        except:
          x[i] = 0
      return x
  elif isinstance(y, list):
    for i in range(len(y)):
      try:
        y[i] = int(x) ^ int(y[i])
      except:
        y[i] = 0
    return y
  else:
    try:
      return int(x) ^ int(y)
    except:
      return 0

def _exp_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      try:
          x[i] = exp(x[i])
      except:
          x[i] = 1
    return x
  else:
    try:
      return exp(x)
    except:
      return 1
      
def _sqrt_(x):
  if isinstance(x, list):
    for i in range(len(x)):
      try:
        x[i] = sqrt(x[i])
      except:
         x[i] = 0
    return x
  else:
    try:
      return sqrt(x)
    except:
      return 0
      
      
def gaussian(x):
  if isinstance(x, list):
    x = sum(x)
  try:
    return exp(-(pow(x,2)/(2.0*pow(.2,2))))
  except OverflowError:
    return 1.0
        
def distance_to_point(x,y,center):
  return sqrt(pow(x-center[0],2) + pow(y-center[1],2)) / sqrt(pow(center[0],2) + pow(center[1],2))

def getcenterdistance(point, img_size):
  x = ((point[0] + 1) * img_size[0]) / 2
  y = ((point[1] + 1) * img_size[0]) / 2
  return distance[int(x), int(y)]
# definition of an Infix operator class
# this recipe also works in jython
# calling sequence for the infix is either:
#  x |op| y
# or:
# x <<op>> y
class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)


_div_ = Infix(protdiv)
_sum_ = Infix(ssum)
_sub_ = Infix(ssub)
_mul_ = Infix(smul)
_xor_ = Infix(sxor)

if __name__=='__main__':
    print (8 |_div_| 2)
    print (9.0 |_div_| 2)
    print (8 |_div_| 0)
    print (8 / 0)
