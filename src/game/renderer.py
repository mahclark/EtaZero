import pygame


class Renderer:

    default_colors = {
        0: (237, 184, 121),
        1: (224, 127, 133),
        2: (128, 90, 91),
        3: (127, 55, 115),
        4: (240, 222, 196),
        5: (63, 61, 55),
        6: (136, 159, 175),
    }

    @staticmethod
    def draw_board(surf, board, selected=None, animated_tiles=None, colors=None):
        if colors == None:
            colors = Renderer.default_colors
        for row in range(board.base):
            for col in range(board.base):
                tile_size = surf.get_size()[0] // board.base
                thickness = 1 if tile_size <= 25 else 2
                if (
                    selected
                    and selected.get((row, col))
                    and (row, col) in board.get_takable()
                ):
                    pygame.draw.rect(
                        surf,
                        [255, 255, 255],
                        (
                            tile_size * col + thickness,
                            tile_size * row + thickness,
                            tile_size - 2 * thickness,
                            tile_size - 2 * thickness,
                        ),
                    )
                    pygame.draw.rect(
                        surf,
                        [0, 0, 0],
                        (
                            tile_size * col + 2 * thickness,
                            tile_size * row + 2 * thickness,
                            tile_size - 4 * thickness,
                            tile_size - 4 * thickness,
                        ),
                    )

                tile = board.get_at(row, col)
                if tile >= 0:
                    pygame.draw.rect(
                        surf,
                        colors[tile],
                        (
                            tile_size * col + 2 * thickness + 1,
                            tile_size * row + 2 * thickness + 1,
                            tile_size - 4 * thickness - 2,
                            tile_size - 4 * thickness - 2,
                        ),
                    )

            if animated_tiles:
                for pos, color, lerp in animated_tiles:
                    pygame.draw.rect(
                        surf,
                        colors[color],
                        (
                            tile_size * pos[1] + 5 + (tile_size / 2 - 5) * lerp,
                            tile_size * pos[0] + 5 + (tile_size / 2 - 5) * lerp,
                            (tile_size - 10) * (1 - lerp),
                            (tile_size - 10) * (1 - lerp),
                        ),
                    )

    @staticmethod
    def draw_score_grid(surf, score, colors=None):
        if colors == None:
            colors = Renderer.default_colors
        grid_width, grid_height = surf.get_size()
        cell_size = int((grid_width - 1) / (2 * score.base + 1))
        for row in range(score.base + 1):
            pygame.draw.line(
                surf,
                [200, 200, 200],
                (0, row * cell_size),
                (cell_size * score.base, row * cell_size),
            )
            pygame.draw.line(
                surf,
                [200, 200, 200],
                (cell_size * (score.base + 1), row * cell_size),
                (grid_width, row * cell_size),
            )
        for col in range(2 * score.base + 2):
            pygame.draw.line(
                surf,
                [200, 200, 200],
                (col * cell_size, 0),
                (col * cell_size, grid_height),
            )
        for i in range(score.base):
            pygame.draw.rect(
                surf,
                [255, 255, 255],
                (
                    (score.base - score.get_score(i)) * cell_size,
                    i * cell_size,
                    cell_size + 1,
                    cell_size + 1,
                ),
            )
            pygame.draw.rect(
                surf,
                colors[i],
                (
                    (score.base - score.get_score(i)) * cell_size + 1,
                    i * cell_size + 1,
                    cell_size - 1,
                    cell_size - 1,
                ),
            )
