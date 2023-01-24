import math

import gym
import pygame
import numpy as np
from gym.vector.utils import spaces


class game2048Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "ansi"], "render_fps": 4}

    def __init__(self, reward_type=0, render_mode=None, observation_mode=1, size=4):
        super(game2048Env, self).__init__()
        self.size = size
        self.window_size = 512

        self.observation_mode = observation_mode

        if observation_mode == 0:
            self.observation_shape = (size, size)
            self.observation_space = spaces.Box(low=np.zeros(self.observation_shape),
                                                high=np.ones(self.observation_shape),
                                                dtype=np.float64)
        if observation_mode == 1:
            self.observation_shape = (size, size, 16)
            self.observation_space = spaces.Box(0, 1, self.observation_shape, dtype=int)


        self.action_space = spaces.Discrete(4)

        self.board = None
        # self.np_random = np.random
        self.reward_type = reward_type
        self.sum_of_merges = 0
        self.nr_of_moved_squares = 0
        self.nr_of_steps = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def set_reward_type(self, reward_type=0):
        self.reward_type = reward_type

    def _get_obs(self, layers=16):
        if self.observation_mode == 0:
            return self.board / (2 ** 11)
        if self.observation_mode == 1:
            representation = 2 ** (np.arange(layers, dtype=int) + 1)

            # layered is the flat board repeated layers times
            layered = np.repeat(self.board[:, :, np.newaxis], layers, axis=-1)

            # Now set the values in the board to 1 or zero depending whether they match representation.
            # Representation is broadcast across a number of axes
            layered = np.where(layered == representation, 1, 0)
            return layered

    def _get_reward(self):
        if np.max(self.board) == 2048:
            return float( 2 ** 14)
        if self.reward_type == 0:
            return np.sum(self.board) / 4096
        if self.reward_type == 1:
            return np.max(self.board) / 4096
        if self.reward_type == 2:
            return self.nr_of_steps
        if self.reward_type == 3:
            return self.sum_of_merges
        if self.reward_type == 4:
            return self.sum_of_merges - self.nr_of_moved_squares
        if self.reward_type == 5:
            if self.sum_of_merges != 0:
                return math.log2(self.sum_of_merges) - self.nr_of_moved_squares
            return self.nr_of_moved_squares

    def reset(self):
        # self.np_random = np.random.seed(0)
        self.nr_of_steps = 0


        self.board = np.zeros((self.size, self.size), dtype=np.int64)
        self._place_random_tiles(count=2)

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.nr_of_moved_squares = 0
        self.sum_of_merges = 0
        self.nr_of_steps += 1

        board_copy = self.slide(action)

        reward = self._get_reward()
        terminated = self.is_done()

        if not terminated:
            if np.array_equal(board_copy, self.board):
                reward = float(- 2 ** 14)
                terminated = True
                if np.max(self.board) == 2048:
                    reward = float( 2 ** 14)
            self.board = board_copy

            if not terminated:
                self._place_random_tiles(count=1)

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, {"board": self.board, "reward": self._get_reward(),
                                                 "highest_tile": np.max(self.board)}

    def render(self, mode="rgb_array"):
        if mode == "human":
            self.render_mode = "human"
            return self._render_frame()
        if self.render_mode == "rgb_array":
            return self._render_frame()
        if self.render_mode == "ansi":
            s = 'Score: {}\n'.format(self._get_reward())
            grid = np.array(self.board)
            s += "{}\n".format(grid)
            return s

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
                    color_scale = math.log2(self.board[x, y])
                    color_scale_bounded = max(0, int(255 - 255 * color_scale / 10))
                    color = (color_scale_bounded, color_scale_bounded, 0)
                    pygame.draw.rect(
                        canvas,
                        color,
                        pygame.Rect(x * pix_square_size, y * pix_square_size, pix_square_size, pix_square_size))
                    self.font.render_to(canvas, (
                        x * pix_square_size + pix_square_size / 3, y * pix_square_size + pix_square_size / 3),
                                        str(self.board[x, y]), (255, 255, 255), size=pix_square_size / 3)

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
        if not self.board.all():
            for _ in range(count):
                a = np.random.randint(self.size, size=1)
                b = np.random.randint(self.size, size=1)
                while self.board[a, b] != 0:
                    a = np.random.randint(self.size, size=1)
                    b = np.random.randint(self.size, size=1)

                # if sum([cell for row in self.board for cell in row]) in (0, 2):
                self.board[a, b] = 2
                # else:
                #     self.board[a][b] = self.board.choice((2, 4))

    def slide(self, action):
        board_copy = self.board.copy()
        board_copy = np.rot90(board_copy, action)
        board_copy = self.slide_down(board_copy)
        board_copy = self.merge_down(board_copy)
        board_copy = self.slide_down(board_copy)
        board_copy = np.rot90(board_copy, 4 - action)
        return board_copy

    def slide_down(self, board, is_test=False):
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
                    if not is_test:
                        self.nr_of_moved_squares += 1

        return board

    def merge_down(self, board, is_test=False):
        nr_of_merges = 0
        for x in range(self.size):
            for y in range(self.size - 2, -1, -1):
                if board[y, x] == 0:
                    continue

                if board[y + 1, x] != 0 and board[y + 1, x] == board[y, x]:
                    board[y + 1, x] = board[y, x] * 2
                    board[y, x] = 0
                    nr_of_merges += 1
                    if not is_test:
                        self.sum_of_merges += board[y + 1, x] * 2

        return board

    def is_done(self):
        copy_board = self.board.copy()
        if not self.board.all():
            return False

        for i in range(4):
            copy_board = np.rot90(copy_board, k=1)
            copy_board = self.slide_down(copy_board, is_test=True)
            copy_board = self.merge_down(copy_board, is_test=True)
            copy_board = self.slide_down(copy_board, is_test=True)
            if not copy_board.all():
                return False

        return True
