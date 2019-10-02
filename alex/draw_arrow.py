#!/usr/bin/python

# module to draw an arrow given some direction information

import pygame
import math

def drawArrow(screen, theta):
    print("draw arrow with angle" + theta)
    width = screen.get_width()
    height = screen.get_height()
    r = min(width, height)/2;
    xCentre = width/2
    yCentre = height/2
    x = xCentre + r*math.cos(theta)
    y = yCentre + r * math.sin(theta)
    pygame.draw.line(screen, pygame.Color(0,0,0), (xCentre, yCentre), (x,y)) 
