#!/usr/bin/python

# module to draw an arrow given some direction information

import pygame
import math

def drawArrowNone(screen):
    dim = 200 
    pygame.draw.rect(screen, pygame.Color(255, 255, 255), [0, 0, dim, dim]) 
    pygame.draw.circle(screen, 0, (int(dim/2), int(dim/2)), int(dim/ 2), 4)

def drawArrow(screen, theta):
    drawArrowNone(screen)

    if theta is None:
        return

    dim = 200
    # print (theta)
    xCentre = dim/2
    yCentre = dim/2
    r = dim/2
    x = xCentre + r * math.cos(theta * math.pi / 180)
    y = yCentre + r * math.sin(-theta * math.pi / 180)
    
    pygame.draw.line(screen, pygame.Color(0,0,0), (xCentre, yCentre), (x,y), 4) 

def drawVoice(screen, left, forward, right, back):
    left_pos = [250, 250]
    forward_pos = [300, 200]
    right_pos = [350, 250]
    back_pos = [300, 300]
    color = pygame.Color(255, 0, 0)
    radius = 10

    pygame.draw.circle(screen, pygame.Color(0, 255, 0), [300, 250], radius)

    pygame.draw.circle(screen, pygame.Color(int(forward * 255), 0, 0), forward_pos, radius)
    pygame.draw.circle(screen, pygame.Color(int(left * 255), 0, 0), left_pos, radius)
    pygame.draw.circle(screen, pygame.Color(int(right * 255), 0, 0), right_pos, radius)
    pygame.draw.circle(screen, pygame.Color(int(back * 255), 0, 0), back_pos, radius)

# Levels
def drawMicLevels(screen, levels):
    rect_height = 10
    rect_width = 50
    rect_x_start = 200
    rect_y_start = 0

    for i in range(len(levels)):
        x_start = rect_x_start + (i * rect_width)
        level_height = math.log(levels[i] + 0.01) * rect_height;
        pygame.draw.rect(screen, pygame.Color(0, 0, 0), [x_start, rect_y_start, rect_width, rect_height])
        pygame.draw.rect(screen, pygame.Color(255, 0, 0), [x_start, rect_y_start, rect_width, level_height])
