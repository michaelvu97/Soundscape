#!/usr/bin/python

# module to draw an arrow given some direction information

import pygame
import math

def drawArrowNone(screen):
    dim = 200 
    pygame.draw.rect(screen, pygame.Color(255, 255, 255), [0, 0, dim, dim]) 
    pygame.draw.circle(screen, 0, (int(dim/2), int(dim/2)), int(dim/ 2), 4)

def drawArrow(screen, theta):
    dim = 200
    print (theta)
    xCentre = dim/2
    yCentre = dim/2
    r = dim/2
    x = xCentre + r * math.cos(theta * math.pi / 180)
    y = yCentre + r * math.sin(-theta * math.pi / 180)
   
    drawArrowNone(screen) 
    pygame.draw.line(screen, pygame.Color(0,0,0), (xCentre, yCentre), (x,y), 4) 

# Levels
def drawMicLevels(screen, levels):
    rect_height = 200
    rect_width = 50
    rect_x_start = 200
    rect_y_start = 0

    for i in range(len(levels)):
        x_start = rect_x_start + (i * rect_width)
        level_height = math.sqrt(levels[i]) * rect_height;
        pygame.draw.rect(screen, pygame.Color(0, 0, 0), [x_start, rect_y_start, rect_width, rect_height])
        pygame.draw.rect(screen, pygame.Color(255, 0, 0), [x_start, rect_y_start, rect_width, level_height])
