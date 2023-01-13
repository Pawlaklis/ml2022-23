import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np


class game2048Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=4):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Box(low=2,
                                            high=2 ** 32,
                                            shape=(self.size, self.size),
                                            dtype=np.int64)

        self.action_space = spaces.Discrete(4)

        self.board = None
        self.np_random = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self.board

    def _get_reward(self):
        return np.max(self.board)

    def reset(self, seed=None, options=None):
        self.np_random = np.random.seed(seed)

        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self._place_random_tiles(count=2)

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):

        self.slide(action)

        terminated = self.is_done()

        if not terminated:
            self._place_random_tiles(count=1)

        observation = self._get_obs()
        reward = self._get_reward()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.font.init()
            pygame.display.init()
            self.font = pygame.freetype.SysFont('Times New Roman', 30)

            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size),
                0, 32
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((128, 0, 64))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        for x in range(self.size):
            for y in range(self.size):
                if self.board[x, y] != 0:
                    pygame.draw.rect(
                        canvas,
                        (255 - self.board[x, y] / 2048 * 255, 255 - self.board[x, y] / 2048 * 255, 0),
                        pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size))
                    self.font.render_to(canvas, (
                    x * pix_square_size + pix_square_size / 3, y * pix_square_size + pix_square_size / 3),
                                        str(self.board[x, y]), (0, 0, 0), size=pix_square_size/3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def _place_random_tiles(self, count):
        for _ in range(count):
            a = self.np_random.integers(3, size=1)
            b = self.np_random.integers(3, size=1)
            while self.board[a, b] != 0:
                a = self.np_random.integers(3, size=1)
                b = self.np_random.integers(3, size=1)

            # if sum([cell for row in self.board for cell in row]) in (0, 2):
            self.board[a, b] = 2
            # else:
            #     self.board[a][b] = self.board.choice((2, 4))

    def slide(self, action):
        changed = False
        self.board = np.rot90(self.board, action)
        changed, self.board = self.slide_down(self.board) or changed
        changed, self.board = self.merge_down(self.board) or changed
        changed, self.board = self.slide_down(self.board) or changed
        self.board = np.rot90(self.board, 4 - action)
        return changed, 0

    def slide_down(self, board):
        changed = False
        for x in range(self.size):
            for y in range(self.size - 2, -1, -1):
                if board[y, x] == 0:
                    continue
                next_y = y
                while next_y + 1 != self.size and board[next_y + 1, x] == 0:
                    next_y += 1

                if next_y != y:
                    board[next_y, x] = board[y, x]
                    board[y, x] = 0
                    changed = True

        return changed, board

    def merge_down(self, board):
        changed = False
        for x in range(self.size):
            for y in range(self.size - 2, -1, -1):
                if board[y, x] == 0:
                    continue

                if board[y + 1, x] != 0 and board[y + 1, x] == board[y, x]:
                    board[y + 1, x] = board[y, x] * 2
                    board[y, x] = 0
                    changed = True
        return changed, board

    def is_done(self):
        copy_board = self.board.copy()

        if not self.board.all():
            return False

        for action in [0, 1, 2, 3]:
            rotated_obs = np.rot90(copy_board, k=action)
            _, updated_obs = self.slide_down(rotated_obs)
            _, updated_obs = self.merge_down(rotated_obs)
            _, updated_obs = self.slide_down(rotated_obs)

        if not updated_obs.all():
            return False

        return True
