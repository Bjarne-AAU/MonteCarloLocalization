

textbox = {}
label = {}

label_fps = {'font-color':(255,255,255) }

canvas = {
    'background-color': (160, 140, 140, 255)
}

button = {
    'autosize': False,
    'font-color':(0,0,0),
    'font-color-over': (0,0,0),
    'font-color-down': (0,0,0),
    'font-color-disabled': (50,50,50),

    'border-width': 1,

    'border-color': (0,0,0),
    'border-color-over': (0,0,0),
    'border-color-down': (0,0,0),
    'border-color-disabled': (0,0,0),

    'bg-color': (150,150,150,200),
    'bg-color-over': (50,50,50,100),
    'bg-color-down': (50,50,50,200),
    'bg-color-disabled': (10,100,100,100),

    'focus-color': (0,0,0,0)
 }

def button_cb(widget, text_on=None, text_off=None):
    widget.value = not widget.value
    if text_on is None or text_off is None: return
    if widget.value: widget.text = text_on
    else: widget.text = text_off
