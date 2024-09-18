import numpy as np
import copy
import torch
from utils_ema.geometry_pose import Pose


class CharucoBoard:

    def __init__(self, board_params, pose=Pose(T=torch.eye(4)), device="cpu"):
        self.params = board_params
        self.grid = self.get_board_grid()
        self.pose = pose
        self.set_device(device)

    def set_device(self, device):
        self.grid = self.grid.to(device)
        self.pose = self.pose.to(device)

    def clone(self):
        ch_new = CharucoBoard(self.params, self.pose.clone())
        return ch_new

    def set_pose(self, pose):
        self.pose = pose
        return self

    def get_board_grid(self):
        grid = torch.tensor(
            [
                [
                    [
                        x * self.params["length_square_real"],
                        y * self.params["length_square_real"],
                        0,
                    ]
                    for x in range(self.params["number_x_square"] - 1)
                    # for x in range(self.params["number_x_square"] - 2, -1, -1)
                ]
                for y in range(self.params["number_y_square"] - 1)
            ]
        ).reshape(
            (self.params["number_x_square"] - 1) * (self.params["number_y_square"] - 1),
            3,
        )
        grid[..., 0] -= grid[-1, 0] / 2
        grid[..., 1] -= grid[-1, 1] / 2

        grid[..., 1] *= -1
        return grid

    def get_board_points(self):
        points = self.grid
        points = points.type(self.pose.location().dtype)
        return (self.pose.rotation() @ points.T).T + self.pose.location()
