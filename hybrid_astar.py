import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from heapq import heappop, heappush
from matplotlib.animation import FuncAnimation

def rotate_submatrix(matrix, position, size, angle):
    submatrix = matrix[position[1]:position[1]+size[1], position[0]:position[0]+size[0]]
    rotated_submatrix = rotate(submatrix, angle, reshape=False, order=0)
    return rotated_submatrix

class HybridAStar:
    def __init__(self, start, goal, grid, grid_resolution, yaw_resolution):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.grid_resolution = grid_resolution
        self.yaw_resolution = yaw_resolution
        self.open_set = []
        # (current cost , x, y, direction)
        heappush(self.open_set, (0, start))
        self.came_from = {}
        self.cost_so_far = {}
        self.came_from[start] = None
        self.cost_so_far[start] = 0

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))

    def search(self):
        while self.open_set:
            current_cost, current = heappop(self.open_set)
            if self.is_goal(current):
                return self.reconstruct_path()

            for next in self.get_neighbors(current):
                new_cost = self.cost_so_far[current] + self.cost(current, next)
                if next not in self.cost_so_far or new_cost < self.cost_so_far[next]:
                    self.cost_so_far[next] = new_cost
                    priority = new_cost + self.heuristic(next, self.goal)
                    heappush(self.open_set, (priority, next))
                    # print('search',(priority, next))
                    self.came_from[next] = current
        return []

    def get_neighbors(self, current):
        neighbors = []
        for d in [-1, 0, 1]:  # three possible steering angles
            for step in range(1, 10):  # step size
                new_yaw = current[2] + d * self.yaw_resolution
                new_x = current[0] + step * self.grid_resolution * np.cos(new_yaw)
                new_y = current[1] + step * self.grid_resolution * np.sin(new_yaw)
                new_pos = (new_x, new_y, new_yaw)
                if self.is_valid(new_pos):
                    neighbors.append(new_pos)
        return neighbors

    def is_valid(self, pos):
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < self.grid.shape[1] and 0 <= y < self.grid.shape[0]:
            return self.grid[y, x] == 0
        return False

    def cost(self, current, next):
        return np.linalg.norm(np.array(current[:2]) - np.array(next[:2]))

    def is_goal(self, pos):
        return np.linalg.norm(np.array(pos[:2]) - np.array(self.goal[:2])) < self.grid_resolution

    def reconstruct_path(self):
        path = []
        current = self.goal
        while current is not None:
            path.append(current)
            current = self.came_from[current]
        path.reverse()
        return path
    
# 시작 및 목표 지점 설정
grid = np.zeros((100, 100))
start = (0, 0, 0)
goal = (50, 50, 0)
grid_resolution = 2.0
yaw_resolution = np.deg2rad(15.0)
planner = HybridAStar(start, goal, grid, grid_resolution, yaw_resolution)
path = planner.search()
print(path)