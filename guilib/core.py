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

# from __future__ import absolute_import

import pygame
from pygame.locals import Rect
from pygame import Surface
from pygame import draw
from pygame import display

import guilib
import utils
import styles

#GENERALS
RESIZING_OFFSET = 3
MIN_SIZE = (30,30)

#COSTANTS
MODE_NORMAL = 0
MODE_ALWAYS_TOP = 1
MODE_ALWAYS_BACK = 2
MODE_DIALOG = 3

#ANCHORING
ANCHOR_NONE = 0
ANCHOR_TOP = 1
ANCHOR_BOTTOM = 2
ANCHOR_RIGHT = 4
ANCHOR_LEFT = 8
ANCHOR_TOPLEFT = ANCHOR_TOP | ANCHOR_LEFT
ANCHOR_TOPRIGHT = ANCHOR_TOP | ANCHOR_RIGHT
ANCHOR_BOTTOMLEFT = ANCHOR_BOTTOM | ANCHOR_LEFT
ANCHOR_BOTTOMRIGHT = ANCHOR_BOTTOM | ANCHOR_RIGHT

#AUTOSIZING
AUTOSIZE_NO = 0
AUTOSIZE_FULL = 1
AUTOSIZE_VERTICAL_ONLY = 2

class GuiCoreException(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)

class _DummyContainer(object):
    "This object is used by internal widgets when they create other widgets with no parent."
    def __init__(self, desktop):
        self.desktop = desktop
        self.size_changed = False

    def add(self, widget):
        widget.desktop = self.desktop

    def nextVertPos(self, spacing=0):
        return (0,spacing)

class Widget(object):
    REFRESH_ON_MOUSE_OVER = True
    REFRESH_ON_MOUSE_DOWN = True
    REFRESH_ON_MOUSE_CLICK = True
    REFRESH_ON_MOUSE_LEAVE = True

    GETS_FOCUS = False

    dynamicAttributes = []

    def __init__(self, parent, style = None, position = 0, size = (100,20), visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT):
        if parent == None:
            raise GuiCoreException("Widget must have a parent.")

        #Logic attributes
        self.mouseover = False
        self.mousedown = False
        self.spacedown = False
        self.mouseclick = False

        #Callbacks
        self.onClick = []
        self.onMouseOver = []
        self.onMouseLeave = []
        self.onMouseDown = []
        self.onGotFocus = []
        self.onLostFocus = []

        self.parent = parent
        self.anchor = anchor
        self.position = (0,0)

        if isinstance(position, int) and self.parent:
            self.position = self.parent.nextVertPos(position)
        else:
            self.position = position

        if isinstance(self.parent, Desktop):
            self.desktop = self.parent
        else:
            self.desktop = self.parent.desktop

        self.size = size
        self.visible = visible
        self.enabled = enabled
        self.style = style

        self.dynamicAttributes = ['style', 'size', 'enabled']

        self.needsRefresh = False

        if self.parent:
            self.parent.add(self)

    def __setattr__(self, attribute, value):
        if attribute in self.dynamicAttributes:
            self.needsRefresh = True

        object.__setattr__(self, attribute, value)

    def update(self, topmost):
        if self.enabled:

            b1,b2,b3 = pygame.mouse.get_pressed()
            self.mouseclick = False

            #Updates informations about mouse
            if not b1:
                if self == topmost and not self.mouseover:
                    self.mouseover = True

                    #Callback!
                    self._runCallbacks(self.onMouseOver)

                    if self.REFRESH_ON_MOUSE_OVER:
                        self.needsRefresh = True

                elif self != topmost and self.mouseover:
                    self.mouseover = False

                    #Callback
                    self._runCallbacks(self.onMouseLeave)

                    if self.REFRESH_ON_MOUSE_LEAVE:
                        self.needsRefresh = True

            #Checks if mouse is over and button is pressed
            if (self.mouseover and b1 or self.hasFocus and pygame.key.get_pressed()[pygame.K_SPACE]) and not self.mousedown:
                #Sets the button is pressed
                self.mousedown = True

                if pygame.key.get_pressed()[pygame.K_SPACE]: self.spacedown = True

                #Refreshes if he wants
                if self.REFRESH_ON_MOUSE_DOWN:
                    self.needsRefresh = True

            #In this case, button is not pressed but the mouse was pressed before
            elif not b1 and self.mousedown and not pygame.key.get_pressed()[pygame.K_SPACE]:
                self.mousedown = False

                #If the mouse is on it too, it means widget has been clicked
                if self.mouseover or self.hasFocus and self.spacedown:

                    self.spacedown = False

                    self.mouseclick = True

                    self._runCallbacks(self.onClick)

                    if self.REFRESH_ON_MOUSE_CLICK:
                        self.needsRefresh = True

            #Calls on mouse down event if present
            self._runCallbacks(self.onMouseDown)

        if self.parent.size_changed:
            self.parent.check_widget_anchoring(self)

        #Refreshes if needed
        if self.needsRefresh:
            self.refresh()
            self.needsRefresh = False

    def refresh(self):
        pass

    def draw(self, surf):
        pass

    def _set_style(self, style):
        self._style = style

    def _runCallbacks(self, callbacks):
        for callback in callbacks:
            callback(self)

    def connect(self, callback_name, function):
        "Adds a callback function to a specific callback list."
        if self.__dict__.has_key(callback_name) and isinstance(self.__dict__[callback_name], list):
            self.__dict__[callback_name].append(function)
            return self
        else:
            raise GuiCoreException("Specified callback is not valid callback.")

    def destroy(self):
        "Removes the widget from its parent's widgets list,"
        self.parent.remove(self)

    def _getFocus(self):
        self._runCallbacks(self.onGotFocus)
        self.needsRefresh = True

    def _loseFocus(self):
        self._runCallbacks(self.onLostFocus)
        self.needsRefresh = True

    def _is_anchored_to(self, anchor_side):
        if self.anchor & anchor_side == anchor_side:
            return 1
        else:
            return 0

    def __set_width(self, value):
        self.size = value,self.size[1]
    def __set_height(self, value):
        self.size = self.size[0], value

    def _get_true_x(self):
        if hasattr(self.parent, "get_child_client_x"):
            return self.parent.get_child_client_x(self)
        else:
            return self.x

    def _set_true_x(self, value):
        if hasattr(self.parent, "set_child_client_x"):
            return self.parent.set_child_client_x(self, value)
        else:
            self.position = value , self.y

    def _get_true_y(self):
        if hasattr(self.parent, "get_child_client_y"):
            return self.parent.get_child_client_y(self)
        else:
            return self.y

    def _set_true_y(self, value):
        if hasattr(self.parent, "set_child_client_y"):
            return self.parent.set_child_client_y(self, value)
        else:
            self.position = self.x, value

    def _set_true_position(self, position):
        self._set_true_x(position[0])
        self._set_true_y(position[1])
    #Properties
    _true_x = property(_get_true_x, _set_true_x)
    _true_y = property(_get_true_y,_set_true_y)
    _true_position = property(lambda self: (self._true_x, self._true_y), _set_true_position)

    rect = property(lambda self: Rect(self._true_position, self.size))
    hasFocus = property(lambda self: self.desktop.focused == self)
    style = property(lambda self: self._style, (lambda self,style: self._set_style(style)))

    width = property(lambda self: self.size[0], __set_width)
    height = property(lambda self: self.size[1], __set_height)
    x = property(lambda self: self.position[0])
    y = property(lambda self: self.position[1])

class Desktop(object):
    def __init__(self):
        self.widgets = []
        self.backWindows = []
        self.windows = []
        self.frontWindows = []
        self.dialogWindows = []

        self.position = (0,0)
        self.focused = None
        self.lastAdded = None

        self.dummy_cont = _DummyContainer(self)

        self.windows_auto_position =  None

        self.blacksurf = Surface((500,500))
        self.blacksurf.fill(0)
        self.blacksurf.set_alpha(200)

        self.size_changed = True


    width = property(lambda self: pygame.display.get_surface().get_width())
    height = property(lambda self: pygame.display.get_surface().get_height())
    size = property(lambda self: pygame.display.get_surface().get_size())
    client_size = property(lambda self: pygame.display.get_surface().get_size())
    dialog_mode = property(lambda self: len(self.dialogWindows) > 0)

    def add(self, widget):
        widget.parent = self

        self.changeFocus(widget)

        if isinstance(widget, Window):
            if widget.mode == MODE_ALWAYS_BACK:
                self.backWindows.append(widget)
            elif widget.mode == MODE_ALWAYS_TOP:
                self.frontWindows.append(widget)
            elif widget.mode == MODE_DIALOG:
                self.dialogWindows.append(widget)
            else:
                self.windows.append(widget)
        else:
            self.lastAdded = widget
            self.widgets.append(widget)

    def remove(self, widget):
        if isinstance(widget, Window):
            if widget.mode == MODE_ALWAYS_BACK:
                self.backWindows.remove(widget)
            elif widget.mode == MODE_ALWAYS_TOP:
                self.frontWindows.remove(widget)
            elif widget.mode == MODE_DIALOG:
                self.dialogWindows.remove(widget)
                if not self.dialog_mode:
                    self.windows_auto_position = None
            else:
                self.windows.remove(widget)
        else:
            self.widgets.remove(widget)

    def update(self):
        "Updates all the GUI system."

        #TAB key handling
        for e in guilib._events:
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_TAB:
                    #TAB pressed
                    if self.focused:
                        widgets = self.focused.parent.widgets

                        index = widgets.index(self.focused) + 1

                        if index == len(widgets):
                            index = 0

                        while not (widgets[index].GETS_FOCUS and widgets[index].enabled and widgets[index].visible):
                            index += 1
                            if index == len(widgets):
                                index = 0

                        self.changeFocus(widgets[index])

        b1,b2,b3 = pygame.mouse.get_pressed() #Obtainse buttons status

        #Topmost is the top most widget under mouse cursor.
        topmost = self.findTopMost()

        #Checks the focused widget
        if b1:
            self.changeFocus(topmost)

        #Lets all contained widgets to update regularly
        for widget in self.widgets + self.backWindows + self.windows + self.frontWindows + self.dialogWindows:
            widget.update(topmost)

    def findTopMost(self):
        point = mx,my = pygame.mouse.get_pos() #Obtains mouse screen coordinates

        topmost = self

        if not self.dialog_mode:
            widget_list = self.widgets + self.backWindows + self.windows + self.frontWindows
        else:
            widget_list = [self.dialogWindows[-1]]

        for widget in widget_list:
            if widget.enabled and widget.visible:
                if widget.rect.collidepoint(point):
                    topmost = widget

        if isinstance(topmost, Container):
            return topmost.findTopMost(topmost._true_position)
        else:
            return topmost

    def draw(self, surf = None):
        if surf == None:
            surf = display.get_surface()

        for widget in self.widgets:
            if widget.visible: widget.draw(surf)
        for window in self.backWindows:
            if window.visible: window.draw(surf)
        for window in self.windows:
            if window.visible: window.draw(surf)
        for window in self.frontWindows:
            if window.visible: window.draw(surf)

        if self.dialog_mode:

            for window in self.dialogWindows[:-1]:
                window.draw(surf)

            utils.drawHVTiled((0,0), (surf.get_size()), self.blacksurf, surf)

            self.dialogWindows[-1].draw(surf)


    def changeFocus(self, widget):
        if self.focused and self.focused != widget:
            self.focused._loseFocus()
            self.focused = None

        if widget != self and widget.GETS_FOCUS and widget != self.focused:
            self.focused = widget
            self.focused._getFocus()

    def bringToFront(self, widget):
        list = None
        if isinstance(widget, Window):
            if widget.mode == MODE_ALWAYS_BACK:
                list = self.backWindows
            elif widget.mode == MODE_ALWAYS_TOP:
                list = self.frontWindows
            elif widget.mode == MODE_NORMAL:
                list = self.windows
            elif widget.mode == MODE_DIALOG:
                return;
        else:
            list = self.widgets

        list.remove(widget)
        list.append(widget)

    def check_widget_anchoring(self, widget):
        if widget._is_anchored_to(ANCHOR_LEFT) and widget._is_anchored_to(ANCHOR_RIGHT):
            if hasattr(widget, 'autosize') and widget.autosize == 1:
                widget._set_true_x = utils.centerXY((0,0), self.client_size, widget.size)[0]
            else:
                widget.size = self.client_size[0] - widget.position[0] * 2 , widget.height

    def nextVertPos(self, spacing = 1):
        "DEPRECATED"
        if not self.lastAdded:
            return (0,spacing)
        else:
            return self.lastAdded.position[0], self.lastAdded.position[1] + self.lastAdded.height + spacing

    def _winAutoPosition(self, size):
        if not self.windows_auto_position:
            self.windows_auto_position = utils.centerXY((0,0), self.size, size)
            return self.windows_auto_position

        x,y = self.windows_auto_position
        if x + size[0] > self.width or y + size[1] > self.height:
            self.windows_auto_position = (20,20)
        else:
            self.windows_auto_position = x + 20, y + 20

        return self.windows_auto_position

class Container(Widget):

    client_rect = property(lambda self: Rect((self._true_x + self.style['offset-top-left'][0], self._true_y + self.style['offset-top-left'][1]), self.client_size))
    client_size = property(lambda self: (self.surf.get_width() - self.style['offset-top-left'][0] - self.style['offset-bottom-right'][0], self.surf.get_height() - self.style['offset-top-left'][1] - self.style['offset-bottom-right'][1]))
    _valid_client_rect = property(lambda self: self.client_size[0] > 0 and self.client_size[1] > 0)

    def __init__(self, parent, style, position = 2, size = (100,50), visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT):
        Widget.__init__(self, parent, style, position, size, visible, enabled,anchor)

        self.widgets = []
        self.surf = None
        self.size_changed = True

        self.lastAdded = None

    def check_widget_anchoring(self, widget):
        "To be ovverrided for custom Containers"
        if hasattr(widget, 'autosize') and widget.autosize != 1 or not hasattr(widget, 'autosize'):
            if widget._is_anchored_to(ANCHOR_LEFT) and widget._is_anchored_to(ANCHOR_RIGHT):
                widget.size = self.client_size[0] - widget.position[0] * 2 , widget.height

    def add(self, widget):
        self.lastAdded = widget
        widget.parent = self
        widget.desktop = self.desktop
        self.widgets.append(widget)

    def remove(self, widget):
        widget.parent = None
        self.widgets.remove(widget)

    def get_child_client_x(self, widget):
        if widget._is_anchored_to(ANCHOR_LEFT) and (not widget._is_anchored_to(ANCHOR_RIGHT) or (hasattr(widget, 'autosize') and widget.autosize != 1)):
            return widget.x

        elif widget._is_anchored_to(ANCHOR_RIGHT) and not widget._is_anchored_to(ANCHOR_LEFT):
            return self.client_size[0] - widget.width - widget.x

        else:
            return self.client_size[0] / 2 - widget.width / 2

    def set_child_client_x(self, widget, value):
        if widget._is_anchored_to(ANCHOR_LEFT):
            widget.position = value, self.y

        elif widget._is_anchored_to(ANCHOR_RIGHT):
            widget.position =  self.client_size[0] - value, widget.y


    def get_child_client_y(self, widget):
        if widget._is_anchored_to(ANCHOR_TOP) and (not widget._is_anchored_to(ANCHOR_BOTTOM) or (hasattr(widget, 'autosize') and widget.autosize != 1)):
            return widget.y

        elif widget._is_anchored_to(ANCHOR_BOTTOM) and not widget._is_anchored_to(ANCHOR_TOP):
            return self.client_size[1] - widget.height - widget.y

        else:
            return self.client_size[1] / 2 - widget.height / 2

    def set_child_client_y(self, widget, value):
        if widget._is_anchored_to(ANCHOR_TOP):
            widget.position = widget.x ,value

        elif widget._is_anchored_to(ANCHOR_BOTTOM):
            widget.position = widget.x, self.client_size[1] - value

    def refresh(self):
        if self.surf != self.size:
            self.size_changed = True

    def draw(self, surf):
        #Supposing method update was called before this, all widgets have redrawn their
        #private surfaces so they are ready to be drawn on the screen.

        #If the widget is cropped by the left or top edge of the screen-container,
        #I copy its surface and let child widgets draw on it, the blit it on screen, otherwise
        #I take the subsurface from the screen and let widgets draw on it which is faster.
        if self._true_x < 0 or self._true_y < 0:
            temp_surf = self.surf.copy()

            if self._valid_client_rect:
                temp_client_surf = self.get_client_subsurf(temp_surf)

                for widget in self.widgets:
                    if widget.visible: widget.draw(temp_client_surf)

            surf.blit(temp_surf, self._true_position)
        else:
            surf.blit(self.surf, self._true_position)

            if self._valid_client_rect:
                subsurfrect = self.client_rect.clip(surf.get_rect())

                if subsurfrect.size != (0,0):
                    subsurf = surf.subsurface(subsurfrect)

                    for widget in self.widgets:
                        if widget.visible: widget.draw(subsurf)

    def update(self, topmost):
        for widget in self.widgets:
            widget.update(topmost)

        self.size_changed = False

        Widget.update(self, topmost)

    def findTopMost(self, screenpos):
        if not self._valid_client_rect:
            return self

        mx,my = pygame.mouse.get_pos() #Obtains mouse screen coordinates

        point = mx - screenpos[0] - self.style['offset-top-left'][0], my - screenpos[1] - self.style['offset-top-left'][1]

        if point[0] >= self.client_size[0] or point[1] >= self.client_size[1]:
            return self

        topmost = self

        for widget in self.widgets:
            if widget.enabled and widget.visible:
                if widget.rect.collidepoint(point):
                    topmost = widget

        if isinstance(topmost, Container) and topmost != self:
            return topmost.findTopMost((topmost._true_x + self.style['offset-top-left'][0] + screenpos[0],
                                        topmost._true_y + self.style['offset-top-left'][1] + screenpos[1]))
        else:
            return topmost

    def nextVertPos(self, spacing = 1):
        if not self.lastAdded:
            return (0,spacing)
        else:
            return self.lastAdded.x, self.lastAdded.y + self.lastAdded.height + spacing

    def get_client_subsurf(self,  surf = None):
           if not surf: surf = self.surf
           return surf.subsurface(Rect(self.style['offset-top-left'], (surf.get_width() - self.style['offset-top-left'][0] - self.style['offset-bottom-right'][0], surf.get_height() - self.style['offset-top-left'][1] - self.style['offset-bottom-right'][1])))

class Canvas(Container):

    REFRESH_ON_MOUSE_OVER  = False
    REFRESH_ON_MOUSE_DOWN = False
    REFRESH_ON_MOUSE_CLICK = False
    REFRESH_ON_MOUSE_LEAVE = False

    def __init__(self, parent, style = None, position = (0,0), size = (100,20), anchor = ANCHOR_TOP | ANCHOR_LEFT):
        s = styles.defaultCanvasStyle.copy()
        if style is not None: s.update(style)
        Container.__init__(self,parent,s,position,size,True,True,anchor)

        self.canvas = pygame.Surface(self.size, pygame.SRCALPHA, 32)
        self.surf = pygame.Surface(self.size)

    def draw(self, surf):
        if 'background-color' in self.style:
            self.surf.fill(self.style['background-color'])
        self.surf.blit(self.canvas, (0,0))
        Container.draw(self, surf)


class Label(Widget):

    REFRESH_ON_MOUSE_OVER  = False
    REFRESH_ON_MOUSE_DOWN = False
    REFRESH_ON_MOUSE_CLICK = False
    REFRESH_ON_MOUSE_LEAVE = False

    def __init__(self, parent, style = None, position = 0, size = (100,20), visible = True, anchor = ANCHOR_TOP | ANCHOR_LEFT, autosize = True, text = "Label"):
        s = styles.defaultLabelStyle.copy()
        if style is not None: s.update(style)
        Widget.__init__(self,parent,s,position,size,visible,False, anchor)

        self.text = text
        self.autosize = autosize

        self.dynamicAttributes.append("text")
        self.refresh()

    def refresh(self):
        self.surf = utils.renderText(self.text, self.style['font'], self.style['antialias'], self.style['font-color'],
                   self.size, self.autosize, self.style['wordwrap'])

        if self.autosize == AUTOSIZE_FULL:
            self.size = self.surf.get_size()
        elif self.autosize == AUTOSIZE_VERTICAL_ONLY:
            self.size = self.size[0], self.surf.get_height()

    def draw(self, surf):
        surf.blit(self.surf, self._true_position, Rect((0,0), self.size))
        if self.style['border-width']:
            draw.rect(surf, self.style['border-color'], self.rect, self.style['border-width'])



class Button(Widget):
    GETS_FOCUS = True
    def __init__(self, parent = None, style = None, position = 0, size = (100,20), visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT, autosize = True, text = "Button", image = None):
        s = styles.defaultButtonStyle.copy()
        if style is not None: s.update(style)
        Widget.__init__(self,parent,s,position,size,visible,enabled,anchor)

        self.surf = None
        self.text = text
        self.image = image
        self.autosize = autosize

        self.dynamicAttributes.append("text")
        self.refresh()

    def _set_style(self, style):
        """
        This method processes the FIXED skin if given to estrapulate single parts from the whole surface.

        Note: Skins are, differently from guilib, fixed, that means it doesn't support half defined skins (only normal and pressed, i.e.)
        """
        self._style = style

        if style and style['appearence'] == styles.GRAPHICAL:
            h = style['skin'].get_height()
            widths = list(style['widths-normal']) + list(style['widths-over']) + list(style['widths-down']) + list(style['widths-disabled']) + list(style['widths-focused'])

            self.images = {'left': style['skin'].subsurface(Rect(0,0,widths[0],h)),
                           'middle': style['skin'].subsurface(Rect(widths[0],0,widths[1],h)),
                           'right': style['skin'].subsurface(Rect(sum(widths[:2]),0,widths[2],h)),

                           'left-over': style['skin'].subsurface(Rect(sum(widths[:3]),0,widths[3],h)),
                           'middle-over': style['skin'].subsurface(Rect(sum(widths[:4]),0,widths[4],h)),
                           'right-over': style['skin'].subsurface(Rect(sum(widths[:5]),0,widths[5],h)),

                           'left-down':style['skin'].subsurface(Rect(sum(widths[:6]),0,widths[6],h)),
                           'middle-down': style['skin'].subsurface(Rect(sum(widths[:7]),0,widths[7],h)),
                           'right-down': style['skin'].subsurface(Rect(sum(widths[:8]),0,widths[8],h)),

                           'left-disabled': style['skin'].subsurface(Rect(sum(widths[:9]),0,widths[9],h)),
                           'middle-disabled': style['skin'].subsurface(Rect(sum(widths[:10]),0,widths[10],h)),
                           'right-disabled': style['skin'].subsurface(Rect(sum(widths[:11]),0,widths[11],h)),

                           'left-focused': style['skin'].subsurface(Rect(sum(widths[:12]),0,widths[12],h)),
                           'middle-focused': style['skin'].subsurface(Rect(sum(widths[:13]),0,widths[13],h)),
                           'right-focused': style['skin'].subsurface(Rect(sum(widths[:14]),0,widths[14],h))}
        else:
            self.images = None

    def refresh(self):
        if not self.enabled:
            suffix = '-disabled'
        elif self.mousedown:
            suffix = "-down"
        elif self.mouseover:
            suffix = "-over"
        else:
            suffix = ""

        self.suffix = suffix

        textsurf = self.style['font'].render(self.text, self.style['antialias'], self.style['font-color'+suffix])

        content_width = 0
        if self.image:
            content_width += self.image.get_width()
        if self.text:
            content_width += textsurf.get_width()
        if self.image and self.text:
            content_width += 2

        if self.autosize:
            if self.style['appearence'] == styles.VECTORIAL:
                if self.image:
                    img_height = self.image.get_height()
                else:
                    img_height = 0
                self.size = content_width + 14, max(textsurf.get_height(), img_height) + 4
            else:


                self.size = 2 + self.images['left'].get_width() + self.images['right'].get_width() + content_width, self.images['middle'].get_height()

        #Then, it creates its brand new surface if size changed or it's the first time it refreshes.
        if not self.surf or self.surf.get_size() != self.size:
            self.surf = Surface(self.size, pygame.SRCALPHA)

        if self.style['appearence'] == styles.VECTORIAL:
            self.surf.fill(self.style['bg-color' + suffix])
            draw.rect(self.surf, self.style['border-color'+self.suffix], Rect((0,0), self.size), self.style['border-width'])
        else:
            #Clears the surface
            self.surf.fill((0,0,0,0))
            utils.drawHWidget((0,0), self.width, self.images['left' + suffix], self.images['middle' + suffix], self.images['right' + suffix], self.surf)

        if self.image:
            if self.text:
                point = utils.centerXY((0,0), self.size, (2 + self.image.get_width() + textsurf.get_width(), max(self.image.get_height() , textsurf.get_height())))

                self.surf.blit(self.image, point)
                self.surf.blit(textsurf, (point[0] + self.image.get_width() + 2, self.height / 2 - textsurf.get_height()  / 2))
            else:
                self.surf.blit(self.image , utils.centerXY((0,0), self.size, self.image.get_size()))
        else:
            self.surf.blit(textsurf, utils.centerXY((0,0), self.size, textsurf.get_size()))

        if self.hasFocus:
            if self.style['appearence'] == styles.VECTORIAL:
                offset = 1
                draw.rect(self.surf, self.style['focus-color'], Rect(offset,offset,self.width-offset*2,self.height-offset*2), 1)
            else:
                utils.drawHWidget((0,0), self.width, self.images['left-focused'], self.images['middle-focused'], self.images['right-focused'], self.surf)

    def draw(self, surf):
        surf.blit(self.surf, self._true_position)


class ImageButton(Widget):
    def __init__(self, parent = None, skin = None, position = 0, visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT):
        Widget.__init__(self,parent,None,position,None,visible,enabled,anchor)

        self.skin = skin

        self.dynamicAttributes.append("skin")
        self.refresh()

    def refresh(self):
        subs = utils.splitCostantSurface(self.skin, 4)

        self.images= {'normal': subs[0],
                      'over':   subs[1],
                      'down':   subs[2],
                      'disabled':   subs[3]
                      }

        self.size =  self.images['normal'].get_size()

    def draw(self,surf):
        if not self.enabled:
            suffix = 'disabled'
        elif self.mousedown:
            suffix = "down"
        elif self.mouseover:
            suffix = "over"
        else:
            suffix = "normal"

        surf.blit(self.images[suffix], self._true_position)

class Window(Container):

    REFRESH_ON_MOUSE_OVER = False
    REFRESH_ON_MOUSE_DOWN = False
    REFRESH_ON_MOUSE_CLICK = False
    REFRESH_ON_MOUSE_LEAVE = False

    GETS_FOCUS = False

    def __init__(self, parent, style = None, position = None, size = (300,200), visible = True, enabled = True, title = "Window", closeable = True, shadeable = True, mode = MODE_NORMAL, resizable = False):
        self.title = title
        self.closeable = closeable
        self.shadeable = shadeable
        self.mode = mode
        self.resizable = resizable

        self.min_size = MIN_SIZE

        def closebutton_click(widget):
            self.close()

        self.shaded = False

        def shadebutton_click(widget):
            if not self.shaded:
                self.shade()
            else:
                self.unshade()


        self.closebutton_click = closebutton_click
        self.shadebutton_click = shadebutton_click

        if not position:
            if mode == MODE_DIALOG:
                position = parent._winAutoPosition(size)
            else:
                position = utils.centerXY((0,0), parent.size, size)


        s = styles.defaultWindowStyle.copy()
        if style is not None: s.update(style)
        Container.__init__(self, parent, s, position, size, visible, enabled, ANCHOR_TOP | ANCHOR_LEFT)

        self.moving = None
        self.resizing = None

        #Callbacks
        self.onMove = []
        self.onMoveStop = []
        self.onResizeEnd = []
        self.onShade = []
        self.onClose = []

        self.dynamicAttributes.extend(('title','mode', 'closeable','shadeable','dialog'))
        self.refresh()

    window_rect = property(lambda self: Rect(self.style['border-offset-top-left'], (self.width - self.style['border-offset-top-left'][0] -
                                                                                    self.style['border-offset-bottom-right'][0],
                                                                                    self.height - self.style['border-offset-top-left'][1] -                                                                               self.style['border-offset-bottom-right'][1])))
    def _set_style(self, style):
        self._style = style

        if style:
            if style['appearence'] == styles.VECTORIAL:
                self.min_size = style['header-offset'] * 2 + style['header-height'] * 3, style['header-offset'] * 2 + style['header-height']

                if self.closeable:
                    self.closebutton = Button(parent = self.desktop.dummy_cont, style = style['close-button-vectorial-style']
                                          , text = 'X').connect('onClick', self.closebutton_click)
                    self.closebutton.size = self.style['header-height'] -4 ,self.style['header-height']-4

                if self.shadeable:
                    self.shadebutton = Button(parent = self.desktop.dummy_cont, style = style['shade-button-vectorial-style']
                                          , text = '-').connect('onClick', self.shadebutton_click)

                    self.shadebutton.size = self.style['header-height'] -4 ,self.style['header-height']-4
            else:
                self.min_size = (max(self.style['image-top-left'].get_width(),self.style['image-bottom-right'].get_width(),self.style['image-left'].get_width()) +
                                 max(self.style['image-top'].get_width(), self.style['image-bottom'].get_width(), self.style['image-middle'].get_width()) +
                                 max( self.style['image-top-right'].get_width(), self.style['image-right'].get_width(), self.style['image-bottom-right'].get_width()),

                                 max(self.style['image-top-left'].get_height(),self.style['image-top'].get_height(),self.style['image-top-right'].get_height()) +
                                 max(self.style['image-bottom-right'].get_height(), self.style['image-bottom'].get_height(), self.style['image-bottom-right'].get_height()) +
                                 max( self.style['image-left'].get_height(), self.style['image-right'].get_height(), self.style['image-middle'].get_height()))

                if self.closeable:
                    self.closebutton = ImageButton(parent = self.desktop.dummy_cont, skin = self.style['close-button-skin']).connect('onClick', self.closebutton_click)
                if self.shadeable:
                    self.shadebutton = ImageButton(parent = self.desktop.dummy_cont, skin = self.style['shade-button-skin']).connect('onClick', self.shadebutton_click)

    def refresh(self):
        Container.refresh(self)
        if not self.surf or self.surf.get_size() != self.size:
            self.surf = Surface(self.size, pygame.SRCALPHA)
        else:
            self.surf.fill((0,0,0,0))

        if self.style['appearence'] == styles.VECTORIAL:
            self.surf.fill(self.style['bg-color'])
            headersurf = Surface( (self.width- self.style['header-offset']*2, self.style['header-height']), pygame.SRCALPHA)
            headersurf.fill(self.style['header-color'])
            self.surf.blit(headersurf, (self.style['header-offset'], self.style['header-offset']))

            draw.rect(self.surf, self.style['border-color'], Rect((0,0), self.size), self.style['border-width'])

            if self.closeable:
                self.closebutton.position = self.width - self.closebutton.width - self.style['header-offset']-2, \
                                    self.style['header-offset']+2
            if self.shadeable:
                self.shadebutton.position = self.width - self.shadebutton.width - self.style['header-offset'] - self.style['header-height'], \
                                    self.style['header-offset']+2
        else:
            utils.drawGraphicWidget(Rect((0,0), self.size), self.surf, self.style['image-top-left'], self.style['image-top'], self.style['image-top-right'],
                                    self.style['image-right'], self.style['image-bottom-right'],
                                     self.style['image-bottom'], self.style['image-bottom-left'], self.style['image-left'],
                                     self.style['image-middle'])

            if self.closeable:
                self.closebutton.position = self.width - self.closebutton.width - self.style['close-button-offset'][0], \
                                        self.style['close-button-offset'][1]

            if self.shadeable and self.closeable:
                self.shadebutton.position = self.width - self.shadebutton.width - self.style['shade-button-offset'][0], \
                                        self.style['shade-button-offset'][1]
            elif self.shadeable:
                self.shadebutton.position = self.width - self.shadebutton.width - self.style['shade-button-only-offset'][0], \
                                        self.style['shade-button-only-offset'][1]
        #Title
        width = self.width
        if self.shadeable and self.shadebutton.x < width:
            width = self.shadebutton.x
        if self.closeable and self.closebutton.x < width:
            width = self.closebutton.x

        self.surf.set_clip(Rect(0,0, width, self.height))

        self.surf.blit(self.style['font'].render(self.title, True, self.style['font-color']), self.style['title-position'])

        self.surf.set_clip()

    def update(self, topmost):
        oldMousedown = self.mousedown

        Container.update(self, topmost)

        if self.closeable: self.closebutton.update(topmost)
        if self.shadeable: self.shadebutton.update(topmost)

        mx,my = pygame.mouse.get_pos()

        if self.moving:
            dx = mx - self.moving[0]
            dy = my - self.moving[1]

            self.position = (self.x + dx, self.y + dy)

            self.moving = pygame.mouse.get_pos()

        if self.resizing:

            dx = mx - self.resizing[1][0]
            dy = my - self.resizing[1][1]

            for side in self.resizing[0]:
                if side == 'L': #Left-side
                    newsize = self.resizing[2][0] - dx, self.size[1]
                    if newsize[0] > self.min_size[0]:
                        if self.shaded:
                            self._oldsize = newsize[0], self._oldsize[1]
                            self.size = newsize[0], self.height
                        else:
                            self.size = newsize
                        self.position = mx - self.style['border-offset-top-left'][0], self.y

                elif side == 'R': #Right-side
                    newsize = self.resizing[2][0] + dx, self.size[1]
                    if newsize[0] > self.min_size[0]:
                        if self.shaded:
                            self._oldsize = newsize[0], self._oldsize[1]
                            self.size = newsize[0], self.height
                        else:
                            self.size = newsize

                elif side == 'T': #Top-side
                    newsize = self.size[0], self.resizing[2][1] - dy
                    if newsize[1] > self.min_size[1]:
                        if self.shaded:
                            self._oldsize = newsize[0], self._oldsize[1]
                            self.size = newsize[0], self.height
                        else:
                            self.size = newsize
                            self.position = self.x, my -  self.style['border-offset-top-left'][1]

                elif side == 'B': #bottom-side
                    newsize = self.size[0], self.resizing[2][1] + dy
                    if newsize[1] > self.min_size[1]:
                        if self.shaded:
                            self._oldsize = newsize[0], self._oldsize[1]
                            self.size = newsize[0], self.height
                        else:
                            self.size = newsize

            #self.needsRefresh = True

        if not oldMousedown and self.mousedown and self.enabled:
            mx,my = pygame.mouse.get_pos()

            sides = ''

            if self.resizable and self.mouseover:
                if self._true_x + self.style['border-offset-top-left'][0] - RESIZING_OFFSET <= mx <= self._true_x + self.style['border-offset-top-left'][0] + RESIZING_OFFSET:
                    sides = 'L'
                elif self._true_x  + self.size[0] - self.style['border-offset-bottom-right'][0] - RESIZING_OFFSET <= mx <= self._true_x + self.size[0] - self.style['border-offset-bottom-right'][0] + RESIZING_OFFSET:
                    sides = 'R'

                if self._true_y + self.style['border-offset-top-left'][1] - RESIZING_OFFSET <= my <= self._true_y + self.style['border-offset-top-left'][1] + RESIZING_OFFSET:
                    sides += 'T'
                elif self._true_y + self.size[1] - self.style['border-offset-bottom-right'][1] - RESIZING_OFFSET <= my <= self._true_y + self.size[1] - self.style['border-offset-bottom-right'][1] + RESIZING_OFFSET:
                    sides += 'B'

            if sides:
                self.resizing = sides, (mx,my), self.size
            else:
                #Start moving
                self.moving = pygame.mouse.get_pos()
                self._runCallbacks(self.onMove)

        elif oldMousedown and not self.mousedown:
            #Stop moving
            if self.moving:
                self.moving = None
                self._runCallbacks(self.onMoveStop)

            #Stop resizing
            if self.resizing:
                self.resizing = None
                self._runCallbacks(self.onResizeEnd)

        if self.mousedown and self.enabled:
            self.parent.bringToFront(self)

    def draw(self, surf):
        Container.draw(self, surf)

        if self.shadeable or self.closeable:
            if self.shadeable:
                sh = self.shadebutton._true_y + self.shadebutton.height
            else:
                sh = 0

            if self.closeable:
                ch = self.closebutton._true_y + self.closebutton.height
            else:
                ch = 0

            h = max(sh, ch)
            sub = Surface((self.width,h), pygame.SRCALPHA)

            if self.closeable: self.closebutton.draw(sub)
            if self.shadeable: self.shadebutton.draw(sub)

            surf.blit(sub, self._true_position)

    def fitToContent(self):
        max_x = 0
        max_y = 0

        for widget in self.widgets:
            max_x = max(max_x, widget.x + widget.width)
            max_y = max(max_y, widget.y + widget.height)


        self.size = (max_x + self.style['offset-top-left'][0] + self.style['offset-bottom-right'][0],
                     max_y + self.style['offset-top-left'][1] + self.style['offset-bottom-right'][1])
        return self

    def findTopMost(self, screenpos):
        mx,my = pygame.mouse.get_pos()
        point = mx - screenpos[0], my - screenpos[1]

        if self.closeable and self.closebutton.rect.collidepoint(point):
            return self.closebutton
        elif self.shadeable and self.shadebutton.rect.collidepoint(point):
            return self.shadebutton
        else:
            return Container.findTopMost(self, screenpos)

    #Window Methods
    def close(self):
        self.destroy()
        self._runCallbacks(self.onClose)

    def shade(self):
        if not self.shaded:
            self.shaded = True
            self._oldsize = self.size

            if self.style['appearence'] == styles.VECTORIAL:
                self.size = self.width, self.style['header-height'] + self.style['header-offset'] * 2
            else:
                self.size = self.width, self.style['image-top'].get_height() + self.style['image-bottom'].get_height()

    def unshade(self):
        if self.shaded:
            self.shaded = False
            self.size = self._oldsize
class TextBox(Widget):

    GETS_FOCUS = True

    REFRESH_ON_MOUSE_OVER = False
    REFRESH_ON_MOUSE_LEAVE = False
    REFRESH_ON_MOUSE_DOWN = False

    def __init__(self, parent = None, style = None, position = 0, size = (100,20), visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT, text = ""):
        s = styles.defaultTextBoxStyle.copy()
        if style is not None: s.update(style)
        Widget.__init__(self,parent,s,position,size,visible,enabled, anchor)

        self.text = text
        self.currpos = len(text)
        self._textStartX = 0
        self.surf = None
        self.textWidth = 0

        pygame.key.set_repeat(250, 40)

        self.dynamicAttributes.extend(["text", "currpos"])
        self.refresh()

    def _set_style(self, style):
        self._style = style

        if style:
            if style['appearence'] == styles.GRAPHICAL:
                subs = utils.splitSurface(style['skin'], *(style['widths-normal'] + style['widths-disabled']))

                self.images= {'left':   subs[0],
                              'middle': subs[1],
                              'right':  subs[2],
                              'left-disabled':   subs[3],
                              'middle-disabled': subs[4],
                              'right-disabled':  subs[5]
                              }
            else:
                self.images = None

    def refresh(self):
        #Save this information coz it's frequently used
        #self.metrics = self.style['font'].metrics(self.text)
        self.offset = self.style['offset']

        if self.size[1] < self.style['font'].get_ascent() + self.offset[1]* 2:
            self.size = self.size[0], self.style['font'].get_ascent() + self.offset[1]* 2

        if not self.enabled:
            suffix = '-disabled'
        elif self.hasFocus:
            suffix = "-focus"
        else:
            suffix = ""

        #Creates the surface with the rendered text
        self.textsurf = self.style['font'].render(self.text, self.style['antialias'], self.style['font-color' + suffix])

        #Creates a new widget surface if None or different size from widget size
        if not self.surf or self.surf.get_size() != self.size:
            self.surf = pygame.Surface(self.size, pygame.SRCALPHA)

        if self.style['appearence'] == styles.VECTORIAL:

            #Background
            self.surf.fill(self.style['bg-color' + suffix])

            #Calculates the position of the text surface
            textpos =  self.offset[0], self.size[1] / 2. - self.textsurf.get_height() / 2

            #Width of the text until the cursor
            cursorWidth = self.style['font'].size(self.text[:self.currpos])[0]
            #X coordinate of the cursor
            cursorx = cursorWidth + self.offset[0]
            #Total width of the text
            self.textWidth = self.textsurf.get_width()

            if cursorWidth - self._textStartX < self.size[0] - self.offset[0] * 2 :
                if cursorx - self._textStartX < 0:
                    self._textStartX = max(0, cursorx - (self.size[0] - self.offset[0] * 2))
            else:
                self._textStartX = cursorWidth - (self.size[0] - self.offset[0] * 2)

            bgcolor = self.style['bg-color' + suffix]
            color_dark = utils.change_color_brightness(bgcolor,-50)
            color_light = utils.change_color_brightness(bgcolor,+30)
            #color_corner = mixColors(color_dark, color_light)

            draw.line(self.surf,color_dark, (0,0), (self.size[0]-2,0)) #TOP
            draw.line(self.surf,color_dark, (0,0), (0,self.size[1]-2)) #LEFT
            draw.line(self.surf,color_light, (1,self.size[1]-1), (self.size[0],self.size[1]-1)) #LEFT
            draw.line(self.surf,color_light, (self.size[0]-1,1), (self.size[0]-1,self.size[1]-1)) #LEFT

        #Blits the text surface in the appropriate position
        self.surf.blit(self.textsurf, textpos, Rect(self._textStartX ,0, self.size[0] - self.offset[0] * 2, self.textsurf.get_height()))

        #Draws the cursor
        if self.hasFocus:
            cursorx -= self._textStartX
            draw.line(self.surf, (255,255,255), (cursorx ,self.offset[1]),(cursorx , self.size[1] - self.offset[1]))

    def update(self, topmost):
        Widget.update(self, topmost)

        #Letter entry
        if self.currpos > len(self.text):
            self.currpos = len(self.text)

        if self.hasFocus and self.enabled:
            for e in guilib._events:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_BACKSPACE:
                        if self.currpos == 0:
                            continue
                        self.text = self.text[:self.currpos-1] + self.text[self.currpos:]
                        self.currpos -= 1
                        if self.currpos < 0:
                            self.currpos = 0
                    elif e.key == pygame.K_DELETE:
                        self.text = self.text[:self.currpos] + self.text[self.currpos+1:]
                    elif e.key == pygame.K_LEFT:
                        self.currpos -= 1
                        if self.currpos < 0:
                            self.currpos = 0
                    elif e.key == pygame.K_RIGHT:
                        self.currpos += 1
                        if self.currpos > len(self.text):
                            self.currpos = len(self.text)
                    elif e.key == pygame.K_HOME:
                        self.currpos = 0
                    elif e.key == pygame.K_END:
                        self.currpos = len(self.text)
                    elif e.key in (pygame.K_RSHIFT, pygame.K_LSHIFT, pygame.K_RETURN, pygame.K_TAB):
                        pass
                    else:
                        self.text = self.text[:self.currpos] +  e.unicode + self.text[self.currpos:]
                        self.currpos += 1

    def draw(self, surf):
        surf.blit(self.surf, self._true_position)

class CheckBox(Widget):
        def __init__(self, parent, style = None, position = 10, size = (100,20), visible = True, enabled = True, anchor = ANCHOR_TOP | ANCHOR_LEFT, autosize = True, text = "Checkbox", value = True):
            s = styles.defaultCheckBoxStyle.copy()
            if style is not None: s.update(style)
            Widget.__init__(self,parent,s,position,size,visible,enabled,anchor)

            self.autosize = autosize
            self.text = text
            self.value = value
            self.textsurf = None
            self.surf = None

            self.dynamicAttributes.extend(["text","value", "autosize"])

            #Callbacks
            self.onValueChanged =  []
            self.refresh()

            self.connect("onClick", self._mouse_click)

        def _mouse_click(self, widget):
            self.value = not self.value
            self._runCallbacks(self.onValueChanged)

        def refresh(self):
            if self.enabled:
                suffix = ""
            else:
                suffix = "-disabled"

            self.textsurf = utils.renderText(self.text, self.style['font'], self.style['antialias'], self.style['font-color'+suffix],
                   self.size, self.autosize, self.style['wordwrap'])

            if not self.enabled:
                suffix = '-disabled'
            elif self.mousedown:
                suffix = "-down"
            elif self.mouseover:
                suffix = "-over"
            else:
                suffix = ""

            if self.value:
                prefix = "checked"
            else:
                prefix = "unchecked"

            if self.autosize:
                if self.style['appearence'] == styles.VECTORIAL:
                    self.size = self.style['box-width'] + self.style['spacing'] + self.textsurf.get_width(), max(self.textsurf.get_height(), self.style['box-width'])
                else:
                    self.size = (self.textsurf.get_width() + self.style['spacing'] + self.style['checked-normal'].get_width(), max (self.textsurf.get_height(), self.style['checked-normal'].get_height()))

            if not self.surf or self.size != self.surf.get_size():
                self.surf = Surface(self.size, pygame.SRCALPHA)
            else:
                self.surf.fill((0,0,0,0))

            if self.style['appearence'] == styles.VECTORIAL:
                text_location = self.style['box-width'] + self.style['spacing'] , self.height / 2 - self.textsurf.get_height() / 2

                rect = Rect(0, 0, self.style['box-width'], self.style['box-width'])

                #Background
                draw.rect(self.surf, self.style['box-color' + suffix], rect)
                #Cross
                if self.value:
                    draw.rect(self.surf, self.style['check-color'+suffix], rect.inflate(-self.style['box-width']/2, -self.style['box-width']/2))

                draw.rect(self.surf, self.style['border-color'+suffix], rect, 1)

                self.surf.blit(self.textsurf, text_location)
            else:
                centerPoint = self.style['spacing']  + self._true_x + self.style['checked-normal'].get_width(), center(self.position, self.size, self.textsurf.get_size())[1]

                image = self.style[prefix + suffix]
                imagePoint = self.position[0], center(self.position,self.size, image.get_size())[1]

                #Draws the image
                surf.blit(image, imagePoint)
                surf.blit(self.textsurf, centerPoint, Rect((0,0), self.size))

        def draw(self, surf):
            surf.blit(self.surf, self._true_position)

