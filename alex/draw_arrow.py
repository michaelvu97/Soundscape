#!/usr/bin/python

# module to draw an arrow given some direction information

import pygame
import math

def drawArrowNone(screen):
    dim = min(screen.get_width(), screen.get_height())
    
    pygame.draw.rect(screen, pygame.Color(255, 255, 255), [0, 0, dim, dim]) 
    pygame.draw.circle(screen, 0, (int(dim/2), int(dim/2)), int(dim/ 2), 4)

def drawArrow(screen, theta):
    print (theta)
    dim = min(screen.get_width(), screen.get_height())
    xCentre = dim/2
    yCentre = dim/2
    r = dim/2
    x = xCentre + r*math.cos(theta * math.pi / 180)
    y = yCentre + r * math.sin(-theta * math.pi / 180)
   
    drawArrowNone(screen) 
    pygame.draw.line(screen, pygame.Color(0,0,0), (xCentre, yCentre), (x,y), 4) 
