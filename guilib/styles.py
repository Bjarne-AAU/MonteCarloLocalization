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
import os

### Style Costants ###
VECTORIAL = 0
GRAPHICAL = 1




### Default Styles ###
path = os.path.dirname(os.path.abspath(__file__))

#Default font
defaultFont = pygame.font.Font(os.path.join(path,"font.otf"), 14)
titleFont = pygame.font.Font(os.path.join(path,"font.otf"), 14)
titleFont.set_bold(True)

# Default Canvas Style
defaultCanvasStyle = {
    'offset-top-left': (0, 0),
    'offset-bottom-right': (0,0),
#    'background-color': (160, 140, 140, 255)
}

#Default Label Style
defaultLabelStyle = {'font': defaultFont,
                     'font-color': (0,0,0),
                     'bg-color': (0,0,0,0),
                     'border-width': 0,
                     'border-color': (0,0,0),
                     'wordwrap': True,
                     'antialias': True}

#Default Button Style
defaultButtonStyle = {'font': defaultFont,
                      'antialias': True,

                      'font-color':(255,255,255),
                      'font-color-over': (0,0,0),
                      'font-color-down': (100,100,100),
                      'font-color-disabled': (100,100,100),

                      'appearence': VECTORIAL,

                      'border-width': 1,

                      'border-color': (0,0,0),
                      'bg-color': (0,0,0,200),

                      'border-color-over': (0,0,0),
                      'bg-color-over': (100,100,100,200),

                      'border-color-down': (0,0,0),
                      'bg-color-down': (0,0,0,250),

                      'border-color-disabled': (0,0,0),
                      'bg-color-disabled': (90,90,90),

                      'focus-color': (0,0,0,100)
                      }

graphicButtonStyleTemplate = {'font': defaultFont,
                              'antialias': True,

                              'font-color':(0,0,0),
                              'font-color-over': (0,0,0),
                              'font-color-down': (0,0,0),
                              'font-color-disabled': (0,0,0),

                              'appearence': GRAPHICAL,

                              'skin': None,
                              'widths-normal': (4,1,4),
                              'widths-over': (4,1,4),
                              'widths-down': (4,1,4),
                              'widths-disabled': (4,1,4),
                              'widths-focused': (0,0,0)
                              }

windowButtonsStyle = defaultButtonStyle.copy()
windowButtonsStyle['autosize'] = False

defaultWindowStyle = {'font': titleFont,
                      'font-color': (255,255,255),
                      'offset-top-left': (7,37),
                      'offset-bottom-right': (7,7),
                      'title-position': (10,8),

                      'appearence': VECTORIAL,

                      'bg-color': (0,0,0,150),
                      'border-width': 1,
                      'border-color': (30,50,100),
                      'header-color': (0,0,50,100),
                      'header-offset': 5,
                      'header-height': 25,
                      'close-button-vectorial-style':  windowButtonsStyle,
                      'shade-button-vectorial-style':  windowButtonsStyle,
                      }



graphicWindowStyleTemplate = {'font': titleFont,
                      'font-color': (255,255,255),
                      'offset-top-left': (7,37),
                      'offset-bottom-right': (7,7),
                      'title-position': (10,8),

                      'appearence': GRAPHICAL,

                      'border-offset-top-left': (0,0),
                      'border-offset-bottom-right': (0,0),

                      'image-top-left': None,
                      'image-top': None,
                      'image-top-right': None,
                      'image-right': None,
                      'image-bottom-right': None,
                      'image-bottom': None,
                      'image-bottom-left': None,
                      'image-left': None,

                      'close-button-skin':  None,
                      'shade-button-skin':  None,

                      'close-button-offset': (5,5),
                      'shade-button-offset': (30,5),
                      'shade-button-only-offset': (5,5)
                      }

defaultTextBoxStyle = {'font': defaultFont,
                      'antialias': True,
                      'offset': (4,4),

                      'font-color':(200,200,255),
                      'font-color-focus': (255,255,255),
                      'font-color-disabled': (0,0,0),

                      'appearence': VECTORIAL,

                      'border-color': (0,0,0),
                      'bg-color': (55,55,55),

                      'border-color-focus': (0,50,50),
                      'bg-color-focus': (70,70,80),

                      'border-color-disabled': (0,0,0),
                      'bg-color-disabled': (90,90,90),
                      }

defaultCheckBoxStyle = {'font': defaultFont,
                        'font-color': (0,0,0),
                        'font-color-disabled':(50,50,50),
                        'wordwrap': True,
                        'antialias': True,
                        'spacing': 4,

                        'appearence': VECTORIAL,

                        'box-width': 15,

                        'box-color': (50,50,50,100),
                        'box-color-over': (150,150,150,200),
                        'box-color-down': (50,50,50,100),
                        'box-color-disabled': (10,100,100,100),

                        'border-color': (0,0,0),
                        'border-color-over': (0,0,0),
                        'border-color-down': (0,0,0),
                        'border-color-disabled': (0,0,0),

                        'check-color': (255,255,255),
                        'check-color-over': (100,100,255),
                        'check-color-down': (5,5,5),
                        'check-color-disabled': (250,250,250)
                        }
