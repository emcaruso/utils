import torch
import numpy as np
import copy


class Quat:
    def __init__(self, params=None, device=None, dtype=None):
        if params is None:
            params = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device="cpu")
        if device is None:
            device = params.device
        if dtype is None:
            dtype = params.dtype
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params)
        self.params = params
        self.device = device

    @classmethod
    def from_rot(cls, R: torch.Tensor):
        # Ensure the rotation matrix is valid
        assert R.shape[-2:] == (3, 3)
        if not torch.allclose(
            R.det(), torch.tensor(1.0, dtype=R.dtype, device=R.device)
        ):
            raise ValueError(
                f"The input matrix {R} is not a valid rotation matrix (det != 1)."
            )
        # Compute the quaternion components
        trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        w = torch.sqrt(torch.clamp(1 + trace, min=1e-6)) / 2

        x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
        y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
        z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
        q = cls(torch.stack((w, x, y, z), dim=-1))
        return q

    def clone(self):
        return Quat(
            copy.deepcopy(self.params.detach().cpu()),
            device=self.device,
            dtype=self.params.dtype,
        )

    def is_normalized(self):
        return torch.allclose(
            torch.norm(self.params, dim=-1),
            torch.tensor(1.0, dtype=self.params.dtype, device=self.params.device),
        )

    def normalize_data(self):
        self.params.data = self.normalized_params().data
        return self

    def normalized_params(self):
        norm = torch.norm(self.params, dim=-1, keepdim=True)
        eps = 1e-6
        if torch.any(norm < eps):  # Avoid dividing by very small numbers
            return self.params / (norm + eps)  # Safeguard against tiny values
        else:
            return self.params / norm

    def to_rot(self):
        """convert the quaternion to a rotation matrix"""

        # Ensure the quaternion is normalized
        quat = self.normalized_params()

        # Extract components
        w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

        # Compute rotation matrix
        R = torch.zeros(quat.shape[:-1] + (3, 3), dtype=quat.dtype, device=quat.device)
        R[..., 0, 0] = 1 - 2 * y * y - 2 * z * z
        R[..., 0, 1] = 2 * x * y - 2 * w * z
        R[..., 0, 2] = 2 * x * z + 2 * w * y
        R[..., 1, 0] = 2 * x * y + 2 * w * z
        R[..., 1, 1] = 1 - 2 * x * x - 2 * z * z
        R[..., 1, 2] = 2 * y * z - 2 * w * x
        R[..., 2, 0] = 2 * x * z - 2 * w * y
        R[..., 2, 1] = 2 * y * z + 2 * w * x
        R[..., 2, 2] = 1 - 2 * x * x - 2 * y * y

        return R

    def to(self, device):
        self.params = self.params.to(device)
        self.device = device
        return self

    # def quat2axis_angle(self):
    #     # Ensure quaternion is normalized
    #     quat = self.params / torch.norm(self.params, dim=-1, keepdim=True)
    #
    #     # Extracting scalar and vector parts
    #     w, v = quat[...,0], quat[...,1:]
    #     angle = 2 * torch.acos(w)
    #
    #     # Ensure the angle is within a valid range
    #     angle = torch.clamp(angle, min=-torch.pi, max=torch.pi)
    #
    #     # Compute rotation axis
    #     sin_theta_2 = torch.sqrt(1 - w * w)
    #     axis = v / sin_theta_2 if sin_theta_2 > 1e-6 else torch.tensor([1.0, 0.0, 0.0])
    #
    #     return axis, angle

    def get_inv(self):
        q_conj = self.params.clone()
        q_conj[..., 1:] = -q_conj[..., 1:]
        return Quat(q_conj)

    def __sub__(self, other):
        q1_inv = self.get_inv()
        return q1_inv * other

    def __mul__(self, other):
        if isinstance(other, Quat):
            new_rot = self.to_rot() @ other.to_rot()
            new_eul = self.from_rot(new_rot)
            return new_eul
        else:
            raise ValueError("Can only multiply by another eul instance")

        # # Extract components
        # w1, x1, y1, z1 = (
        #     self.params[..., 0],
        #     self.params[..., 1],
        #     self.params[..., 2],
        #     self.params[..., 3],
        # )
        # w2, x2, y2, z2 = (
        #     other.params[..., 0],
        #     other.params[..., 1],
        #     other.params[..., 2],
        #     other.params[..., 3],
        # )
        #
        # # Compute components of the product
        # w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        # x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        # y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        # z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        #
        # return Quat(torch.stack((w, x, y, z), dim=-1))

    def dist(self, q2):
        q1_inv = self.get_inv()
        q_dist = q2 * q1_inv
        # q_dist = self*q2
        _, angle = q_dist.quat2axis_angle()
        return angle
