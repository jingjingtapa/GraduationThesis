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

###### 이미지 전체 크기 ######
#차선 폭 : 15cm, 차로 폭 : 3m, 여백 : 15cm
margin, line_width = 15, 15
lane_width, car_width = int(line_width * 20), int(line_width * 12.7)
car_height = int(car_width*2.63)
width, height = 5*line_width+2*lane_width, 2000
grid = np.zeros((height, width))

###### 차선 ######
left_line, center_line, right_line = int(margin), int(2*margin + lane_width), int(width - margin - line_width)

# 차선을 그리기
grid[:, left_line:left_line + line_width] = 1
grid[:, right_line:right_line + line_width] = 1
        
###### 주변 차량 ######
# 좌하측 위치, 너비, 높이
other_car = [(85, 50), (85, 700), (85,1300)]
cnt = 2
for cx, cy in other_car:
    grid[cy:cy+car_height+1, cx:cx+car_width+1] = cnt
    cnt += 1

# 시작 및 목표 지점 설정
start = (85, 50, 0)
goal = (1500, 700, 0)
grid_resolution = 2.0
yaw_resolution = np.deg2rad(15.0)
planner = HybridAStar(start, goal, grid, grid_resolution, yaw_resolution)
path = planner.search()



# 경로가 비어있을 경우 처리
# if not path:
#     print("경로를 찾을 수 없습니다.")
# else:
#     # 애니메이션 설정
#     fig, ax = plt.subplots()
#     im = ax.imshow(grid, cmap='gray')

#     def update(frame):
#         grid_copy = grid.copy()
#         pos = path[frame]
#         x, y, yaw = int(pos[0]), int(pos[1]), pos[2]
#         rotated_vehicle = rotate_submatrix(grid_copy, (x, y), (car_width, car_height), np.rad2deg(yaw))
#         x_end = min(x + car_width, grid_copy.shape[1])
#         y_end = min(y + car_height, grid_copy.shape[0])
#         grid_copy[y:y_end, x:x_end] = rotated_vehicle[:y_end-y, :x_end-x]
#         im.set_data(grid_copy)
#         return [im]

#     ani = FuncAnimation(fig, update, frames=len(path), interval=100, blit=True)
#     plt.show()