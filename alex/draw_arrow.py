#!/usr/bin/python

# module to draw an arrow given some direction information

import pygame
import math

def drawArrow(screen, theta):
    width = screen.get_width()
    xCentre = width/2
    yCentre = width/2
    r = width/2
    x = xCentre + r*math.cos(theta * math.pi / 180)
    y = yCentre + r * math.sin(-theta * math.pi / 180)
    pygame.draw.line(screen, pygame.Color(0,0,0), (xCentre, yCentre), (x,y)) 
