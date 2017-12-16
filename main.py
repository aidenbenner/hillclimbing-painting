import random as random
import numpy as np

import matplotlib.pyplot as plt
from skimage import io 
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve, set_color)
from skimage.transform import rescale, resize, downscale_local_mean

i = io.imread("monalisa.jpg")
original = resize(i, (int(i.shape[0] / 8), int(i.shape[1] / 8)))

current = original.copy()
rr, cc = ellipse(0, 0, 4000, 4000, current.shape)
set_color(current, (rr, cc), (0,0,0))

plt.show()
# fig.show()

XMAX = int(original.shape[0])
YMAX = int(original.shape[1])

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return -err

class circle:
    def __init__(self, x, y, ra, rb, col):
        self.x = x
        self.y = y
        self.ra = ra
        self.rb = rb
        self.col = col 

    def circFromTup(self, t):
        return circle(t[0], t[1], t[2], t[3], (t[4], t[5], t[6]))

    def mutate(self):
        a = self.getTuple() 
        b = getRandomCircle(XMAX, YMAX).getTuple()
        i = random.randint(0, len(a) - 1)
        a[i] = round((a[i] + b[i]) / 2)
        self.setFromTuple(a)

    def getChild(self, female):
        # position[a] colors[b] 
        return circle(self.x, self.y, self.ra, self.rb, female.col)

    def getTuple(self):
        return [self.x, self.y, self.ra, self.rb, self.col[0], self.col[1], self.col[2]]

    def setFromTuple(self, t):
        self.x = t[0]
        self.y = t[1]
        self.ra = t[2]
        self.rb = t[3]
        self.col = (t[4], t[5], t[6])

    def hillclimb(self, curr, target): 
        maxes = [XMAX, YMAX, XMAX, YMAX, 255, 255, 255]

        currFit = cfitness(self, curr, target)
        a = self.getTuple()
        for x in range(len(a)): 

            top = self.getTuple()
            bot = self.getTuple()

            topFit = currFit
            while top[x] < maxes[x]:
                # test up 
                top[x] = top[x] + 2
                newFitness = cfitness(self.circFromTup(top), curr, target)
                if newFitness > topFit:
                    topFit = newFitness
                else:
                    top[x] = top[x] - 2
                    break

            botFit = currFit
            while bot[x] > 0:
                # test up 
                bot[x] = bot[x] - 2
                newFitness = cfitness(self.circFromTup(bot), curr, target)
                if newFitness > botFit:
                    botFit= newFitness
                else:
                    bot[x] = bot[x] + 2
                    break

            if(botFit > topFit) :
                self.setFromTuple(bot)
            else:
                self.setFromTuple(top)


def drawCircle(img, circle):
    # fill ellipse
    col = (circle.col[0] / 256.0, circle.col[1] / 256.0 , circle.col[2] / 256.0)
    rr, cc = ellipse(circle.x, circle.y, circle.rb, circle.rb, current.shape)
    set_color(img, (rr, cc), col)


def cordInCirc(circ, x, y):
    if circ.ra == 0 or circ.rb == 0 : return False
    return ((x - circ.x) ** 2 / (circ.ra ** 2)) - ((y - circ.y) ** 2 / (circ.rb ** 2)) <= 1


def cfitness(circ, im1, im2): 
    tmp = im1.copy() 
    drawCircle(tmp, circ)
    return mse(tmp, im2)

# def cfitness(circ, im1px, im2px): 
    # oldsum = 0
    # sum = 0 
    # for x in range(XMAX):
        # for y in range(YMAX):

            # #if not cordInCirc(circ, x, y):
            # #    continue
            # # would be setting 
            # p1 = im1px[(x,y)]
            # p2 = im2px[(x,y)]
            # for z in range(3):
                # v = abs(p1[z] - p2[z])
                # oldsum += v * v
                # v = abs(p2[z] - circ.col[z])
                # sum += v  * v
    # return oldsum - sum 

# def sqdiff(im1, im2):
    # assert(im1.height == im2.height)
    # assert(im1.width == im2.width)

    # width = im1.width
    # height = im1.height
    
    # sum = 0
    # im1px = im1.load()
    # im2px = im2.load()
    # for x in range(width):
        # for y in range(height):
            # p1 = im1px[(x,y)]
            # p2 = im2px[(x,y)]

            # for z in range(3):
                # v = p1[z] - p2[z]
                # sum += v * v

    # return sum

# print(sqdiff(im, original))
# print(sqdiff(original, original))

def getRandomColor(): 
    return (random.randint(0,255), random.randint(0,255), random.randint(0,255))


def getRandomCircle(xmax, ymax): 
    x = random.randint(0,xmax)
    y = random.randint(0,ymax)
    ra = random.randint(0, int(20))
    rb = random.randint(0, int(20))
    # ra = random.randint(0, int(xmax / 2.0))
    # rb = random.randint(0, int(ymax / 2.0))
    c = getRandomColor()
    return circle(x,y, ra,rb,c)


def evaluate(circle, current, target):
    # current, target are the pixel arrays
    return cfitness(circle, current, target) 

def getBest(population, current, target):
    fitness = [ (x, evaluate(x, current, target)) for x in population]
    fitness = sorted(fitness, key = lambda x: x[1])
    return fitness[0]

def evolvePop(population, current, target, clen=8, plen=8, pmutate=0.31, prandom=0.1): 
    fitness = [ (x, evaluate(x, current, target)) for x in population]
    fitness = sorted(fitness, key = lambda x: x[1])
    # lower is better
    parents = [fitness[x][0] for x in range(plen)]
    print("fitness ", fitness[0][1])

#    tmp = current.copy()
#    drawCircle(tmp, fitness[0][0])
#    tmp.show()

    
    for x in range(plen):
        if prandom > random.random():
            parents.append(getRandomCircle(XMAX, YMAX))

    for x in parents:
        if prandom > random.random():
            x.mutate()

    children = []
    while len(children) < clen:
        male = random.randint(0,len(parents) - 1)
        female = random.randint(0,len(parents) - 1)
        if male == female: 
            continue
        else:
            children.append(parents[male].getChild(parents[female]))


    parents.extend(children)
    return parents

def hcpop(pop, current, target):
    fitness = [ (x, evaluate(x, current, target)) for x in population]
    fitness = sorted(fitness, key = lambda x: x[1])
    best = fitness[-1]
    print(best)
    best[0].hillclimb(current, target)
    print(best[1])
    return best[0]

currIm = current.copy()
numCircles = 1000  


for y in range(numCircles):
    population = [getRandomCircle(XMAX, YMAX) for x in range(100)]
    # for x in range(epochs):
        # print("generation ", x)
        # population = evolvePop(population, currentPix, originalPix)
    best = hcpop(population, currIm, original)
    drawCircle(currIm, best)
    print("circle ", y)

io.imshow(currIm)
plt.show()

io.imsave("output.png", currIm)




