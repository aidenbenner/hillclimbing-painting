import random as random
import numpy as np

import matplotlib.pyplot as plt
from skimage import io
from skimage.draw import (line, polygon, circle,
                          circle_perimeter,
                          ellipse, ellipse_perimeter,
                          bezier_curve, set_color)
from skimage.transform import rescale, resize, downscale_local_mean

input_image = io.imread("starrynight.jpg")

SCALE_FACTOR = 4
original = resize(input_image, (int(input_image.shape[0] / SCALE_FACTOR), int(input_image.shape[1] / SCALE_FACTOR)))
output_image = resize(input_image, (int(input_image.shape[0] / 1.0), int(input_image.shape[1] / 1.0)))

current = original.copy()
rr, cc = ellipse(0, 0, 4000, 4000, current.shape)
set_color(current, (rr, cc), (0,0,0))

rr, cc = ellipse(0, 0, 4000, 4000, output_image.shape)
set_color(output_image, (rr, cc), (0,0,0))

io.imshow(output_image)
plt.show()

# rr, cc = ellipse(20 * SCALE_FACTOR, 20 * SCALE_FACTOR, 200 * SCALE_FACTOR, 200 * SCALE_FACTOR, output_image.shape)
# set_color(output_image, (rr, cc), (0,0,0))


XMAX = int(original.shape[0])
YMAX = int(original.shape[1])

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return -err

class circle:
    def clone(self):
        return circle(self.x, self.y, self.ra, self.rb, self.col)

    def __init__(self, x, y, ra, rb, col):
        self.x = x
        self.y = y
        self.ra = ra
        self.rb = rb
        self.col = col

    def circFromTup(self, t):
        return circle(t[0], t[1], t[2], t[3], (t[4], t[5], t[6], t[7]))

    def mutate(self):
        maxes = [XMAX, YMAX, XMAX, YMAX, 255, 255, 255, 255]
        a = self.getTuple()
        ind = np.random.randint(0, len(a))
        # a[ind] = np.random.normal(a[ind], 20, 1)[0]
        a[ind] = np.random.normal(a[ind], 30, 1)[0]
        a[ind] = max(a[ind], 0)
        a[ind] = min(a[ind], maxes[ind])
        self.setFromTuple(a)





    def getChild(self, female):
        # position[a] colors[b]
        return circle(self.x, self.y, self.ra, self.rb, female.col)

    def getTuple(self):
        return [self.x, self.y, self.ra, self.rb, self.col[0], self.col[1], self.col[2], self.col[3]]

    def setFromTuple(self, t):
        self.x = t[0]
        self.y = t[1]
        self.ra = t[2]
        self.rb = t[3]
        self.col = (t[4], t[5], t[6], t[7])

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

    def hc2(self, curr, target):
        maxes = [XMAX, YMAX, XMAX, YMAX, 255, 255, 255, 255]
        currFit = cfitness(self, curr, target)

        newFit = currFit
        while newFit > currFit:
            newFit = currFit
            a = self.getTuple()
            maxState = a
            for x in range(a):
                tmp = list(a)
                tmp[x] += 2
                tmp[x] = min(maxes[x], tmp[x])
                f = cfitness(tmp, curr, target)
                if f > newFit:
                    newFit = f
                    maxState = tmp

            for x in range(a):
                tmp = list(a)
                tmp[x] -= 2
                tmp[x] = max(0, tmp[x])
                f = cfitness(tmp, curr, target)
                if f > newFit:
                    newFit = f
                    maxState = tmp
            currFit = newFit

            self.setFromTuple(maxState)







def drawCircle(img, circle, scale = 1):
    # fill ellipse
    col = (circle.col[0] / 256.0, circle.col[1] / 256.0 , circle.col[2] / 256.0)
    rr, cc = ellipse(circle.x * scale, circle.y * scale, circle.ra * scale, circle.rb * scale, img.shape)
    set_color(img, (rr, cc), col, alpha=circle.col[3] / 256.0)


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
    return (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))


def getRandomCircle(xmax, ymax, xrmax = 200, yrmax = 200):
    x = np.random.randint(0,xmax)
    y = np.random.randint(0,ymax)
    ra = np.random.randint(0, int(yrmax))
    rb = np.random.randint(0, int(xrmax))
    # ra = numpy.randint(0, int(xmax / 2.0))
    # rb = numpy.randint(0, int(ymax / 2.0))
    c = getRandomColor()
    return circle(x,y, ra,rb,c)


def evaluate(circle, current, target):
    # current, target are the pixel arrays
    return cfitness(circle, current, target)

def getBest(population, current, target):
    fitness = [ (x, evaluate(x, current, target)) for x in population]
    fitness = sorted(fitness, key = lambda x: x[1])
    return fitness[0]

def evolvePop(population, current, target, clen=20, plen=5, pmutate=1, prandom=0):
    fitness = [ (x, evaluate(x, current, target)) for x in population]
    fitness = sorted(fitness, key = lambda x: x[1])
    # higher is better
    parents = [fitness[-plen:][x][0] for x in range(plen)]
    # print("fitness ", fitness[-1][1])

#    tmp = current.copy()
#    drawCircle(tmp, fitness[0][0])
#    tmp.show()


    for x in range(plen):
        if prandom > random.random():
            parents.append(getRandomCircle(XMAX, YMAX))

    children = []
    while len(children) < clen:
        for x in parents:
            tmp = x.clone()
            if pmutate > random.random():
                tmp.mutate()
                children.append(tmp)

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
    best[0].hc2(current, target)
    print(best[1])
    return best[0]

currIm = current.copy()
numCircles = 1000
epochs = 40
for y in range(numCircles):
    population = [getRandomCircle(XMAX, YMAX, max(XMAX, 21), max(YMAX, 21)) for x in range(50)]
    # for x in range(epochs):
        # print("generation ", x)
     #   population = evolvePop(population, currIm, original)

    best = hcpop(population, currIm, original)
    drawCircle(currIm, best)
    drawCircle(output_image, best, SCALE_FACTOR)
    io.imsave("output/c" + str(y) + ".png", output_image)
    io.imsave("output/o" + str(y) + ".png", currIm)
    print("circle ", y)

io.imshow(output_image)
plt.show()

io.imsave("output.png", output_image)




