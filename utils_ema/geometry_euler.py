import torch
import copy
import numpy as np
from utils_ema.geometry_quaternion import Quat


class eul:
    def __init__(self, params=None, convention="YXZ", device="cpu"):
        if params is None:
            params = torch.zeros([3])
        if isinstance(params, np.ndarray):
            params = torch.from_numpy(params)
        self.params = params
        self.convention = convention
        self.device = device

    def to(self, device):
        self.params = self.params.to(device)
        self.device = device
        return self

    def to_quat(self):
        return Quat.from_rot(self.to_rot())

    def clone(self):
        return eul(
            copy.deepcopy(self.params.detach().cpu()),
            convention=self.convention,
            device=self.device,
        )

    def normalize_angles(self):
        return (self.params + torch.pi) % (2 * torch.pi) - torch.pi

    def rotate_by_R(self, R):
        # new_rot = R @ self.eul2rot()
        new_rot = self.eul2rot() @ R
        new_eul = self.rot2eul(new_rot)
        self.params = new_eul.params

    def change_convention(self, convention):
        if convention == self.convention:
            return self
        if convention == "YXZ":
            if self.convention == "XYZ":
                self.params = self.rot2eul_YXZ(self.eul2rot_XYZ())
            elif self.convention == "ZYX":
                self.params = self.rot2eul_YXZ(self.eul2rot_ZYX())
        elif convention == "XYZ":
            if self.convention == "YXZ":
                self.params = self.rot2eul_XYZ(self.eul2rot_YXZ())
            elif self.convention == "ZYX":
                self.params = self.rot2eul_XYZ(self.eul2rot_ZYX())
        elif convention == "ZYX":
            if self.convention == "YXZ":
                self.params = self.rot2eul_ZYX(self.eul2rot_YXZ())
            elif self.convention == "XYZ":
                self.params = self.rot2eul_ZYX(self.eul2rot_XYZ())
        else:
            raise ValueError("Unknown euler convention")
        self.convention = convention
        return self

    def eul2quat_YXZ(self):
        c1 = torch.cos(self.params[..., 1] / 2.0)
        c2 = torch.cos(self.params[..., 0] / 2.0)
        c3 = torch.cos(self.params[..., 2] / 2.0)
        s1 = torch.sin(self.params[..., 1] / 2.0)
        s2 = torch.sin(self.params[..., 0] / 2.0)
        s3 = torch.sin(self.params[..., 2] / 2.0)

        # Quaternion components based on YXZ rotation sequence
        w = c1 * c2 * c3 - s1 * s2 * s3
        x = s1 * s2 * c3 + c1 * c2 * s3
        y = s1 * c2 * c3 + c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3

        return Quat(torch.tensor([w, x, y, z]))

    def eul2quat_XYZ(self):
        c1 = torch.cos(self.params[..., 0] / 2.0)
        c2 = torch.cos(self.params[..., 1] / 2.0)
        c3 = torch.cos(self.params[..., 2] / 2.0)
        s1 = torch.sin(self.params[..., 0] / 2.0)
        s2 = torch.sin(self.params[..., 1] / 2.0)
        s3 = torch.sin(self.params[..., 2] / 2.0)

        # Quaternion components based on XYZ rotation sequence
        w = c1 * c2 * c3 - s1 * s2 * s3
        x = s1 * c2 * c3 + c1 * s2 * s3
        y = c1 * s2 * c3 - s1 * c2 * s3
        z = c1 * c2 * s3 + s1 * s2 * c3

        return Quat(torch.tensor([w, x, y, z]))

    def eul2quat_ZYX(self):
        c1 = torch.cos(self.params[..., 0] / 2.0)
        c2 = torch.cos(self.params[..., 1] / 2.0)
        c3 = torch.cos(self.params[..., 2] / 2.0)
        s1 = torch.sin(self.params[..., 0] / 2.0)
        s2 = torch.sin(self.params[..., 1] / 2.0)
        s3 = torch.sin(self.params[..., 2] / 2.0)

        # Quaternion components based on ZYX rotation sequence
        w = c1 * c2 * c3 + s1 * s2 * s3
        x = s1 * c2 * c3 - c1 * s2 * s3
        y = c1 * s2 * c3 + s1 * c2 * s3
        z = c1 * c2 * s3 - s1 * s2 * c3

        return Quat(torch.tensor([w, x, y, z]))

    def eul2quat(self):
        if self.convention == "YXZ":
            return self.eul2quat_YXZ()
        elif self.convention == "XYZ":
            return self.eul2quat_XYZ()
        elif self.convention == "ZYX":
            return self.eul2quat_ZYX()

    def rotate_by_euler(self, params):
        self.rotate_by_R(params.eul2rot())

    def get_cs(self):
        c = torch.cos(self.params)
        s = torch.sin(self.params)
        c1 = c[..., 0]
        c2 = c[..., 1]
        c3 = c[..., 2]
        s1 = s[..., 0]
        s2 = s[..., 1]
        s3 = s[..., 2]
        return c1, c2, c3, s1, s2, s3

    def eul2rot_XYZ(self):
        c1, c2, c3, s1, s2, s3 = self.get_cs()
        r11 = c2 * c3
        r12 = -c2 * s3
        r13 = s2
        r21 = c1 * s3 + c3 * s1 * s2
        r22 = c1 * c3 - s1 * s2 * s3
        r23 = -c2 * s1
        r31 = s1 * s3 - c1 * c3 * s2
        r32 = c3 * s1 + c1 * s2 * s3
        r33 = c1 * c2
        r1 = torch.cat(
            (r11.unsqueeze(-1), r12.unsqueeze(-1), r13.unsqueeze(-1)), dim=-1
        )
        r2 = torch.cat(
            (r21.unsqueeze(-1), r22.unsqueeze(-1), r23.unsqueeze(-1)), dim=-1
        )
        r3 = torch.cat(
            (r31.unsqueeze(-1), r32.unsqueeze(-1), r33.unsqueeze(-1)), dim=-1
        )
        rot = torch.cat((r1.unsqueeze(-1), r2.unsqueeze(-1), r3.unsqueeze(-1)), dim=-1)
        rot = rot.transpose(-2, -1)
        rot = rot.to(self.params.device)
        return rot

    def eul2rot_ZYX(self):
        c1, c2, c3, s1, s2, s3 = self.get_cs()
        r11 = c1 * c2
        r12 = c1 * s2 * s3 - c3 * s1
        r13 = s1 * s3 + c1 * c3 * s2
        r21 = c2 * s1
        r22 = c1 * c3 + s1 * s2 * s3
        r23 = c3 * s1 * s2 - c1 * s3
        r31 = -s2
        r32 = c2 * s3
        r33 = c2 * c3
        r1 = torch.cat(
            (r11.unsqueeze(-1), r12.unsqueeze(-1), r13.unsqueeze(-1)), dim=-1
        )
        r2 = torch.cat(
            (r21.unsqueeze(-1), r22.unsqueeze(-1), r23.unsqueeze(-1)), dim=-1
        )
        r3 = torch.cat(
            (r31.unsqueeze(-1), r32.unsqueeze(-1), r33.unsqueeze(-1)), dim=-1
        )
        rot = torch.cat((r1.unsqueeze(-1), r2.unsqueeze(-1), r3.unsqueeze(-1)), dim=-1)
        rot = rot.transpose(-2, -1)
        rot = rot.to(self.params.device)
        return rot

    def eul2rot_YXZ(self):
        c1, c2, c3, s1, s2, s3 = self.get_cs()
        r11 = c1 * c3 + s1 * s2 * s3
        r12 = c3 * s1 * s2 - c1 * s3
        r13 = c2 * s1
        r21 = c2 * s3
        r22 = c2 * c3
        r23 = -s2
        r31 = c1 * s2 * s3 - c3 * s1
        r32 = c1 * c3 * s2 + s1 * s3
        r33 = c1 * c2
        r1 = torch.cat(
            (r11.unsqueeze(-1), r12.unsqueeze(-1), r13.unsqueeze(-1)), dim=-1
        )
        r2 = torch.cat(
            (r21.unsqueeze(-1), r22.unsqueeze(-1), r23.unsqueeze(-1)), dim=-1
        )
        r3 = torch.cat(
            (r31.unsqueeze(-1), r32.unsqueeze(-1), r33.unsqueeze(-1)), dim=-1
        )
        rot = torch.cat((r1.unsqueeze(-1), r2.unsqueeze(-1), r3.unsqueeze(-1)), dim=-1)
        rot = rot.transpose(-2, -1)
        rot = rot.to(self.params.device)
        return rot

    def eul2rot_YX(self):  # Y=azimuth X=elevation (no roll)
        c = torch.cos(self.params)
        s = torch.sin(self.params)
        c1 = c[..., 0]
        c2 = c[..., 1]
        s1 = s[..., 0]
        s2 = s[..., 1]
        r11 = c1
        r12 = s1 * s2
        r13 = c2 * s1
        r21 = 0
        r22 = c2
        r23 = -s2
        r31 = -s1
        r32 = c1 * s2
        r33 = c1 * c2
        rot = torch.FloatTensor([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
        return rot

    def rot2eul(self, R=torch.eye(3)):
        if self.convention == "YXZ":
            return eul.rot2eul_YXZ(R)
        elif self.convention == "XYZ":
            return eul.rot2eul_XYZ(R)
        elif self.convention == "ZYX":
            return eul.rot2eul_ZYX(R)

    def eul2rot(self):
        # match self.convention:
        #     case "YXZ":
        #         return self.eul2rot_YXZ()
        #     case _:
        #         raise ValueError("Wrong euler convention")
        if self.convention == "YXZ":
            return self.eul2rot_YXZ()
        elif self.convention == "XYZ":
            return self.eul2rot_XYZ()
        elif self.convention == "ZYX":
            return self.eul2rot_ZYX()
        elif self.convention == "YX":
            return self.eul2rot_YX()

    def to_rot(self):
        return self.eul2rot()

    @staticmethod
    def rot2eul_YXZ(R=torch.eye(3)):
        e1 = torch.atan2(R[..., 0, 2], R[..., 2, 2])
        e2 = torch.asin(-R[..., 1, 2])
        e3 = torch.atan2(R[..., 1, 0], R[..., 1, 1])
        e_flat = torch.cat(
            (e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)), dim=-1
        )
        euler = eul(e_flat)
        return euler

    @staticmethod
    def rot2eul_XYZ(R=torch.eye(3)):
        e1 = torch.atan2(-R[..., 1, 2], R[..., 2, 2])
        e2 = torch.asin(R[..., 0, 2])
        e3 = torch.atan2(-R[..., 0, 1], R[..., 0, 0])
        e_flat = torch.cat(
            (e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)), dim=-1
        )
        euler = eul(e_flat)
        return euler

    @staticmethod
    def rot2eul_ZYX(R=torch.eye(3)):
        e1 = torch.atan2(R[..., 1, 0], R[..., 0, 0])
        e2 = torch.asin(-R[..., 2, 0])
        e3 = torch.atan2(R[..., 2, 1], R[..., 2, 2])
        e_flat = torch.cat(
            (e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)), dim=-1
        )
        euler = eul(e_flat)
        return euler

    @classmethod
    def from_rot(cls, R: torch.Tensor, convention="YXZ"):
        assert R.shape[-2:] == (3, 3)
        assert torch.allclose(
            R.det(), torch.tensor(1.0, dtype=R.dtype, device=R.device)
        )
        if convention == "YXZ":
            return cls.rot2eul_YXZ(R)
        elif convention == "XYZ":
            return cls.rot2eul_XYZ(R)
        elif convention == "ZYX":
            return cls.rot2eul_ZYX(R)

    def dist(self, other):
        q1 = self.eul2quat()
        q2 = other.eul2quat()
        dist = q1.dist(q2)
        return dist

    @staticmethod
    def is_rotation_matrix(R) -> bool:
        R = R.type(torch.float32)
        Rt = R.transpose(-2, -1)
        shouldBeIdentity = torch.eye(3).to(R.device)
        # identity is close to R^T @ R, with a precision of 1e-6
        identity = torch.allclose(Rt @ R, shouldBeIdentity, rtol=1e-6, atol=1e-6)

        det = torch.det(R)
        is_rot = identity and (det > 0)
        if not is_rot:
            print(Rt @ R, det)
        return is_rot
