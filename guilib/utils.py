"""
Copyright (c) 2008 Canio Massimo "Keebus" Tristano

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

import pygame
from pygame.locals import Rect
from math import ceil

def wrapText(text, font, width):
    "Splits the given text in multiple lines to fit the given maximum width."
    if width == 0: return ""
    lineWidth = font.size(text)[0]
    if lineWidth > width:
        words = text.split(' ')
        i = 1
        while i < len(words):
            currLine = ' '.join(words[:-i]) 
            if font.size(currLine)[0] <= width:
                return currLine + "\n" + str(wrapText(' '.join(words[len(words)-i:]),font,width))
            i += 1
    else:
        return text

def renderText(text, font, antialias, color, size, autosize, wordwrap):
    "Renders the text with specified parameters and returns the surface containing it."
    oldLines = text.split('\n')
    
    lines = []
    
    #Wordwrapping
    if wordwrap and autosize != 1:
        for line in oldLines:
            lines.extend(wrapText(line, font, size[0]).split('\n'))
    else:
        lines = oldLines
        
    #Text Rendering       
    if len(lines) == 1:
        #Single line text
        return font.render(text, antialias, color)
        
    else:
        #Multi line text
        lineHeight = font.get_linesize()
        
        height = lineHeight * len(lines)
        
        width = 0
        lineSurfs = []
        
        for line in lines:
            linesurf = font.render(line, antialias, color)
            lineSurfs.append(linesurf) 
            if linesurf.get_width() > width:
                width = linesurf.get_width()
        
        surf = pygame.Surface((width,height), pygame.SRCALPHA)
        
        for i in xrange(len(lineSurfs)):
            surf.blit(lineSurfs[i], (0,i * lineHeight))
    
        return surf

def centerXY(positionHost, sizeHost, sizeClient):
    return (positionHost[0] + sizeHost[0] / 2 - sizeClient[0] / 2,
            positionHost[1] + sizeHost[1] / 2 - sizeClient[1] / 2)

def splitSurface(surf, *widths):
    subs = []
    
    height = surf.get_height()
    x = 0
    for width in widths:
        subs.append( surf.subsurface(Rect(x,0,width,height)))
        x += width
    
    return subs
    
def splitCostantSurface(surf, numpieces):
    w = surf.get_width() // numpieces
    
    return splitSurface(surf, *([w] * numpieces))

def drawHWidget(fromPosition, width, left_surf, middle_surf, right_surf, dest):
    x,y = fromPosition
    dest.blit(left_surf, fromPosition)
    dest.blit(right_surf, (x + width - right_surf.get_width(),y))
    drawHTiled((x + left_surf.get_width(), y), x+width-right_surf.get_width(), middle_surf, dest)


def drawHTiled(fromPosition, toXCoordinate, source, dest):
    width = toXCoordinate - fromPosition[0]
    
    if width > 0:
        dest.set_clip(Rect(fromPosition,(width, source.get_height())))
        
        for i in xrange(width // source.get_width() + 1 ):
            dest.blit(source, (fromPosition[0] + i * source.get_width(), fromPosition[1]))
        
        dest.set_clip()
         
def drawVTiled(fromPosition, toYCoordinate, source, dest):
    height = toYCoordinate - fromPosition[1] 
    
    if height > 0:
        dest.set_clip(Rect(fromPosition,(source.get_width(), height)))
    
        for i in xrange(height // source.get_height() + 1):
            dest.blit(source, (fromPosition[0] , fromPosition[1] + i * source.get_height()))
        
        dest.set_clip()
        
def drawHVTiled(fromPosition, toPosition, source, dest):
    width = toPosition[0] - fromPosition[0]
    height = toPosition[1] - fromPosition[1]
    
    if width > 0 and height > 0:
        surf = dest.subsurface(Rect(fromPosition, (width, height)))
        
        for i in xrange(width // source.get_width() + 1):
            for j in xrange(height // source.get_height() +1):
                surf.blit(source, (i * source.get_width(), j * source.get_height()))
        
def drawGraphicWidget(rect, surf, surf_topleft, surf_top, surf_topright, surf_right, surf_bottomright, surf_bottom, surf_bottomleft, surf_left, surf_middle):
    
    surf.blit(surf_topleft, rect.topleft)
    surf.blit(surf_topright, (rect.topright[0] - surf_topright.get_width(), rect.topright[1]) )
    surf.blit(surf_bottomleft, (rect.bottomleft[0], rect.bottomleft[1] - surf_bottomleft.get_height()))
    surf.blit(surf_bottomright, (rect.bottomright[0] - surf_bottomright.get_width(), rect.bottomright[1] - surf_bottomright.get_height()))
    
    drawHTiled( (rect.topleft[0] + surf_topleft.get_width(), rect.topleft[1]), rect.topright[0] - surf_topright.get_width(), surf_top, surf)
    drawHTiled( (rect.bottomleft[0] + surf_bottomleft.get_width(), rect.bottomleft[1] - surf_bottomleft.get_height()), rect.bottomright[0] - surf_bottomright.get_width(), surf_bottom, surf)
    drawVTiled( (rect.topleft[0],rect.topleft[1] + surf_topleft.get_height()), rect.bottomleft[1] - surf_bottomright.get_height(), surf_left, surf  )
    drawVTiled( (rect.topright[0] - surf_topright.get_width(), rect.topright[1] + surf_topright.get_height()), rect.bottomright[1] - surf_bottomright.get_height(), surf_right, surf  )
    
    drawHVTiled( (rect.topleft[0] + surf_topleft.get_width(), rect.topleft[1] + surf_topleft.get_height()), (rect.bottomright[0] - surf_bottomright.get_width(), rect.bottomright[1] - surf_bottomright.get_height()),
                 surf_middle, surf)

#Colors
def change_color_brightness(color, delta):
    if len(color) == 3:
        r,g,b = color
        a = 255
    elif len(color) == 4:
        r,g,b,a = color
    else:
        raise Exception("Colors must be 3 or 4 dimensions tuple.")
        
    r += delta
    g += delta
    b += delta
    
    if r < 0 : r = 0
    if g < 0: g = 0
    if b < 0 : b = 0
    if r > 255: r = 255
    if g> 255: g = 255
    if b> 255: b = 255
    
    return r,g,b,a