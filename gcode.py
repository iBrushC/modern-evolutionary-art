import numpy as np
import pylineclip

TEMPLATE = '''
G21 ;metric
G90 ;absolute position
G28 X0 Y0 ;move X/Y to min endstops
G28 Z0 ;move Z to min endstops
G0 F9000

;Put drawing message on the LCD screen
M117 Drawing {NAME}...

;Scale: {SCALE}
;Offset: {OFFSET}

{LINE_CODE}

G0 X0 Y0 Z10
'''

def clip_points(start, end, mins, maxes) -> tuple:
    x3, y3, x4, y4 = pylineclip.cohensutherland(
        xmin=mins[0],
        ymin=mins[1],
        xmax=maxes[0],
        ymax=maxes[1],
        x1=start[0],
        y1=start[1],
        x2=end[0],
        y2=end[1],
    )
    return ((x3, y3), (x4, y4))

# Should add some algorithm to sort by tip to tail to speed up drawing
def generate_gcode_from_lines(points: np.ndarray, mode: str, name: str, offset: tuple=(0, 0, 0), scale: tuple=(0, 0)) -> None:
    lines = points.reshape((-1, 2, 2))
    line_commands = ""

    for line in lines:
        start = None
        end = None

        # Because lines are being stored in two different ways we need to be able to parse both
        # [[x1, y1], [x2, y2]]
        if mode == "points":
            start = (
                line[0, 0] * scale[0] + offset[0], 
                line[0, 1] * scale[1] + offset[1]
            )
            end = (
                line[1, 0] * scale[0] + offset[0], 
                line[1, 1] * scale[1] + offset[1]
            )

        # [[x, y], [rotation, length]]
        elif mode == "center":
            center = (line[0, 0], line[0, 1])
            displacement = (
                np.cos(line[1, 0] * 2*np.pi) * line[1, 1] * 0.25, 
                np.sin(line[1, 0] * 2*np.pi) * line[1, 1] * 0.25
            )  
            start = (
                (center[0] + displacement[0]) * scale[0] + offset[0],
                (center[1] + displacement[1]) * scale[1] + offset[1]
            )
            end = (
                (center[0] - displacement[0]) * scale[0] + offset[0],
                (center[1] - displacement[1]) * scale[1] + offset[1]
            )
            
        else: break

        start, end = clip_points(start, end, offset, (scale[0] + offset[0], scale[1] + offset[1]))
        start = np.round(np.array(start), 3)
        end = np.round(np.array(end), 3)
        command = f'''
G0 X{start[0]} Y{start[1]}
G0 Z{0 + offset[2]}
G0 X{end[0]} Y{end[1]}
G0 Z{2 + offset[2]}
'''
        line_commands += command
    
    formatted_code = TEMPLATE.replace("{NAME}", name).replace("{SCALE}", str(scale)).replace("{OFFSET}", str(offset)).replace("{LINE_CODE}", line_commands)
    with open(f"{name}.gcode", "w") as output:
        output.write(formatted_code)


# Converts GCODE to lines
def create_lines_from_gcode(filepath: str) -> np.ndarray:
    output = []
    with open(filepath, "r") as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i == (len(lines) - 1): break
            if not line.startswith("G0 X"): continue

            components = line.strip().split()
            x = float(components[1][1:])
            y = float(components[2][1:])
            output.extend([x, y])

    return np.array(output)