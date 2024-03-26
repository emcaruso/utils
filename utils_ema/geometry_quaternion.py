import torch
import numpy as np

class Quat():
    def __init__(self, q=torch.zeros([4]), device='cpu'):
        if isinstance(q, np.ndarray):
            q = torch.from_numpy(q)
        self.q = q
        self.device = device

    def quat2axis_angle(self):
        # Ensure quaternion is normalized
        quat = self.q / torch.norm(self.q)
        
        # Extracting scalar and vector parts
        w, v = quat[0], quat[1:]
        angle = 2 * torch.acos(w)

        # Ensure the angle is within a valid range
        angle = torch.clamp(angle, min=-torch.pi, max=torch.pi)

        # Compute rotation axis
        sin_theta_2 = torch.sqrt(1 - w * w)
        axis = v / sin_theta_2 if sin_theta_2 > 1e-6 else torch.tensor([1.0, 0.0, 0.0])

        return axis, angle

    def get_inv(self):
        q_conj = self.q.clone()
        q_conj[..., 1:] = -q_conj[..., 1:]
        return Quat(q_conj)

    def __mul__(self, other):
        # Extract components
        w1, x1, y1, z1 = self.q[..., 0], self.q[..., 1], self.q[..., 2], self.q[..., 3]
        w2, x2, y2, z2 = other.q[..., 0], other.q[..., 1], other.q[..., 2], other.q[..., 3]

        # Compute components of the product
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quat(torch.stack((w, x, y, z), dim=-1))

    def dist(self, q2):
        q1_inv = self.get_inv()
        q_dist = q2*q1_inv
        # q_dist = self*q2
        _, angle = q_dist.quat2axis_angle()
        return angle
