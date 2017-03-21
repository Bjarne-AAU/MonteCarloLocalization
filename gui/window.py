import os.path
import pygame

import guilib
import guilib.core as widget

import styles



class MainWindow(object):

    def __init__(self, width=800, height=600, name="Window title", fps=25):
        self._size = (width, height)
        self._screen = pygame.display.set_mode(self._size)
        pygame.display.set_caption(name)

        self.FPS = fps
        self.setup()

        self.running = True

    @property
    def size(self): return self._size

    @property
    def width(self): return self._size[0]

    @property
    def height(self): return self._size[1]

    def get_events(self):
        events = guilib.setEvents()
        for e in events:
            if e.type == pygame.KEYUP:
                if e.key == pygame.K_ESCAPE:
                    self.running = False
                # if e.key == pygame.K_F8:
                #     if self._screen.get_flags() & pygame.FULLSCREEN:
                #         pygame.display.set_mode(self._size)
                #     else:
                #         pygame.display.set_mode(self._size, pygame.FULLSCREEN)
            if e.type == pygame.QUIT:
                self.running = False

        return events

    def setup(self):
        self._clock = pygame.time.Clock()

        self._desktop = widget.Desktop()

        # MAIN MAP
        self._mainmap = widget.Canvas(self._desktop, None, position = (0,0), size = (self.width-200, self.height))
        self._fps = widget.Label(self._mainmap, styles.label_fps, position=(5,5), text = "FPS: 0.00")


        # MINI MAP
        self._minimap = widget.Canvas(self._desktop, position = (self.width-200+1, 0), size = (200,200))


        # MENU
        self._menu = widget.Canvas(self._desktop, styles.canvas, position = (self.width-200+1,201), size = (200,self.height-200))

        ## MENU - CONTROLS
        controls = {
            "start" : widget.Button(self._menu, styles.button, position=(5,5), size=(60, 20), text = "Start", autosize = False, anchor = widget.ANCHOR_TOPLEFT).connect('onClick', lambda w: styles.button_cb(w, "Stop", "Start")),
            "step" : widget.Button(self._menu, styles.button, position=(70,5), size=(60, 20), text = "Step", autosize = False, anchor = widget.ANCHOR_TOPLEFT).connect('onClick', styles.button_cb),\
            "reset" : widget.Button(self._menu, styles.button, position=(135,5), size=(60, 20), text = "Reset", autosize = False, anchor = widget.ANCHOR_TOPLEFT).connect('onClick', styles.button_cb),\
        }
        controls["start"].value = False
        controls["step"].value = False
        controls["reset"].value = False

        world = {
            "world_label" : widget.Label(self._menu, styles.label, position=(5,42), size=(40, 15), text = "World"),
            "world" : widget.TextBox(self._menu, styles.textbox, position=(50,40), size=(145, 12), text = "MY_EARTH"),
            "seed_label" : widget.Label(self._menu, styles.label, position=(5,68), size=(45, 15), text = "Seed"),
            "seed" : widget.TextBox(self._menu, styles.textbox, position=(50,65), size=(50, 12), text = "0"),
            "generate" : widget.Button(self._menu, styles.button, position=(5,65), size=(90, 20), text = "Generate", autosize = False, anchor = widget.ANCHOR_TOPRIGHT).connect('onClick', styles.button_cb),
        }
        world["generate"].value = True


        ## MENU - Options
        checkboxes = {
            "hide_robot" : widget.CheckBox(self._menu, position=(5,145), text = "Hide robot"),
            "show_likelihood" : widget.CheckBox(self._menu, position=(5,165), text = "Show likelihood"),
            "show_posterior" : widget.CheckBox(self._menu, position=(5,185), text = "Show posterior"),
            "option4" : widget.CheckBox(self._menu, position=(5,205), text = "Option 4"),
            "option5" : widget.CheckBox(self._menu, position=(5,225), text = "Option 5")
        }


        self._options = {}
        self._options.update(controls)
        self._options.update(world)
        self._options.update(checkboxes)

    def get_option(self, name):
        if hasattr(self._options[name], 'value'):
            return self._options[name].value
        elif hasattr(self._options[name], 'text'):
            return self._options[name].text
        else: return None


    def set_option(self, name, value):
        if hasattr(self._options[name], 'value'):
            self._options[name].value = value
        elif hasattr(self._options[name], 'text'):
            self._options[name].text = str(value)

    def reset_option(self, name):
        self.set_option(name, False)


    def update(self):
        self._clock.tick(self.FPS)

        self._fps.text = "FPS: {0:.2f}".format(self._clock.get_fps())

        self._desktop.update()
        self._desktop.draw()
        pygame.display.flip()


    def get_mainmap_canvas(self):
        return self._mainmap.canvas

    def get_minimap_canvas(self):
        return self._minimap.canvas

    def get_menu_canvas(self):
        return self._menu.canvas



