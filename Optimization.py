import numpy as np
from scipy.optimize import minimize

def calcVolume(x):
    length = x[0]
    height = x[1]
    width = x[2]
    volume = length * height * width
    return volume

def calcSurface(x):
    length = x[0]
    height = x[1]
    width = x[2]
    surface = 2 * length * height + 2 * length * width + 2 * height * width
    return surface

def objective(x):
    return -calcVolume(x)

def constraint(x):
    return 10 - calcSurface(x)

cons = {'type': 'ineq', 'fun':constraint}

lengthGuess = 1
widthGuess = 1
heightGuess = 1

initials = np.array([lengthGuess, widthGuess, heightGuess])

sol = minimize(objective, initials,method='SLSQP', constraints=cons,options={'disp':True})

xOpt = sol.x
volumeOpt = -sol.fun

surfaceOpt =calcSurface(xOpt)

print('Lenght: ' + str(xOpt[0]))
print('Height: ' + str(xOpt[1]))
print('width: ' + str(xOpt[2]))
print('volume: ' + str(volumeOpt))
print('surface: ' + str(surfaceOpt))