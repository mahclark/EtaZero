import colorsys
import win32gui
import win32ui
from ctypes import windll
from game.sevn import Game, State, Board, Score
from math import sqrt
from PIL import Image
from queue import PriorityQueue


def get_lonely_screen_id():
    lonely_screen_ids = []

    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            if win32gui.GetWindowText(hwnd).find("LonelyScreen") != -1:
                lonely_screen_ids.append(hwnd)

    win32gui.EnumWindows(winEnumHandler, None)

    biggest = None
    area = -1
    for hwnd in lonely_screen_ids:
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]

        if w * h > area:
            area = w * h
            biggest = hwnd

    return biggest


def get_colour_at(x, y, dc):
    col = int(win32gui.GetPixel(dc, round(x), round(y)))
    res = (col & 0xFF), ((col >> 8) & 0xFF), ((col >> 16) & 0xFF)
    return res


def get_phone_rect(hwnd):
    x = None
    y = None
    width = None
    height = None

    lonely_rect = win32gui.GetWindowRect(hwnd)
    dc = win32gui.GetWindowDC(hwnd)
    win_width = lonely_rect[2] - lonely_rect[0]
    # win_height = lonely_rect[3] - lonely_rect[1]
    middle = int(win_width / 2)

    y = 10
    while get_colour_at(middle, y, dc) in [(0, 0, 0), (255, 255, 255)]:
        y += 1

    edge_cols = [
        (0, 0, 0),
        (16, 16, 16),
        (17, 17, 17),
        (52, 52, 52),
        (77, 77, 77),
        (255, 255, 255),
    ]

    x = middle
    while get_colour_at(x, y, dc) not in edge_cols:
        x -= 1
    x += 1

    width = middle - x
    while get_colour_at(x + width, y, dc) not in edge_cols:
        width += 1

    height = 0
    while get_colour_at(middle, y + height, dc) != (0, 0, 0) or height < 100:
        height += 1

    return (x, y, width, height)


def get_starting_state():
    """
    Returns the starting state of a game played on the iOS app Sevn.
    The iOS device must be an iPhone 8 screen sharing to LonelyScreen running on Windows 10.
    Also returns the tile colours and background gradient colours.
    """
    hwnd = get_lonely_screen_id()
    dc = win32gui.GetWindowDC(hwnd)
    x, y, width, height = get_phone_rect(hwnd)

    ref_points = [
        (0.5, 0.15),
        (0.5, 0.17),
        (0.5, 0.19),
        (0.5, 0.21),
        (0.5, 0.23),
        (0.5, 0.25),
        (0.5, 0.27),
    ]
    tile_points = [
        (0.425, 0.469),
        (0.451, 0.467),
        (0.538, 0.498),
        (0.625, 0.527),
        (0.711, 0.539),
        (0.799, 0.549),
        (0.943, 0.594),
        (0.316, 0.485),
        (0.405, 0.51),
        (0.49, 0.537),
        (0.58, 0.568),
        (0.664, 0.577),
        (0.75, 0.589),
        (0.836, 0.605),
        (0.268, 0.53),
        (0.355, 0.553),
        (0.442, 0.579),
        (0.531, 0.607),
        (0.615, 0.619),
        (0.704, 0.634),
        (0.789, 0.651),
        (0.221, 0.577),
        (0.307, 0.598),
        (0.394, 0.621),
        (0.48, 0.649),
        (0.567, 0.664),
        (0.655, 0.679),
        (0.742, 0.697),
        (0.174, 0.604),
        (0.262, 0.629),
        (0.344, 0.651),
        (0.435, 0.679),
        (0.522, 0.693),
        (0.609, 0.708),
        (0.693, 0.725),
        (0.122, 0.629),
        (0.211, 0.654),
        (0.298, 0.683),
        (0.385, 0.712),
        (0.471, 0.724),
        (0.557, 0.734),
        (0.645, 0.751),
        (0.131, 0.69),
        (0.163, 0.685),
        (0.25, 0.716),
        (0.337, 0.746),
        (0.423, 0.757),
        (0.51, 0.767),
        (0.59, 0.792),
    ]

    def scale_hsv(col):
        # scale by extra 10 so V of HSV is prioritised less (i.e. shading partially ignored)
        return (col[0], col[1], col[2] / 2550)

    rgb_cols = [
        get_colour_at(x + p[0] * width, y + p[1] * height, dc) for p in ref_points
    ]

    true_cols = [scale_hsv(colorsys.rgb_to_hsv(*col)) for col in rgb_cols]
    index_map = {col: i for i, col in enumerate(true_cols)}

    tile_cols = [
        scale_hsv(
            colorsys.rgb_to_hsv(*get_colour_at(x + p[0] * width, y + p[1] * height, dc))
        )
        for p in tile_points
    ]
    col_map = {(i % 7, i // 7): col for i, col in enumerate(tile_cols)}

    """
    Below is an algorithm to group the tiles into 7 distinct colours (defined by true_cols).
    The tiles are assigned groups greedily by proximity to the true colours of the groups with availiable spots.
    """
    q = PriorityQueue()

    groups = [set() for _ in range(7)]
    group_map = {}

    def dist(v1, v2):
        s = 0
        for a, b in zip(v1, v2):
            s += (a - b) ** 2
        return sqrt(s)

    def get_closest(col):
        dist_list = [
            (dist(col, ref_col), ref_col)
            for ref_col in true_cols
            if len(groups[index_map[ref_col]]) < 7
        ]
        if len(dist_list) == 0:
            print(groups)
            raise Exception("Unable to find a matching - all groups full")
        return min(dist_list)

    for pos, col in col_map.items():
        d, closest = get_closest(col)
        q.put((d, pos, closest))

    while not q.empty():
        d, pos, ref_col = q.get()
        group = index_map[ref_col]

        if len(groups[group]) == 7:
            d, closest = get_closest(col_map[pos])
            q.put((d, pos, closest))
        else:
            groups[group].add(pos)
            group_map[pos] = group

    board = [[group_map[(x, y)] for x in range(7)] for y in range(7)]

    return (
        State(Board(7, board), Score(7), next_go=1),
        rgb_cols,  # Tile colours
        # Background colour (top)
        get_colour_at(x, y, dc),
        # Bakcground colour (bottom)
        get_colour_at(x, y + height * 0.8, dc),
    )
