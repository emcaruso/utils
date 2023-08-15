import torch

class eul():
    def __init__(self, e=torch.zeros([3]), convention="YXZ"):
        self.e = e
        self.convention = convention

    def eul2rot(self):
        # match self.convention:
        #     case "YXZ":
        #         return self.eul2rot_YXZ()
        #     case _:
        #         raise ValueError("Wrong euler convention")
        if self.convention == "YXZ":
            return self.eul2rot_YXZ()
        elif self.convention == "YX":
            return self.eul2rot_YX()

    def get_cs(self):
        c = torch.cos(self.e)
        s = torch.sin(self.e)
        c1 = c[...,0]
        c2 = c[...,1]
        c3 = c[...,2]
        s1 = s[...,0]
        s2 = s[...,1]
        s3 = s[...,2]
        return c1,c2,c3,s1,s2,s3

    def eul2rot_YXZ(self):
        c1,c2,c3,s1,s2,s3 = self.get_cs()
        r11 = c1*c3+s1*s2*s3
        r12 = c3*s1*s2-c1*s3
        r13 = c2*s1
        r21 = c2*s3
        r22 = c2*c3
        r23 = -s2
        r31 = c1*s2*s3-c3*s1
        r32 = c1*c3*s2+s1*s3
        r33 = c1*c2
        r1 = torch.cat( (r11.unsqueeze(-1), r12.unsqueeze(-1), r13.unsqueeze(-1)), dim =-1)
        r2 = torch.cat( (r21.unsqueeze(-1), r22.unsqueeze(-1), r23.unsqueeze(-1)), dim =-1)
        r3 = torch.cat( (r31.unsqueeze(-1), r32.unsqueeze(-1), r33.unsqueeze(-1)), dim =-1)
        rot = torch.cat( (r1.unsqueeze(-1), r2.unsqueeze(-1), r3.unsqueeze(-1)), dim =-1)
        rot = rot.to(self.e.device)
        return rot
        
    def eul2rot_YX(self): # Y=azimuth X=elevation (no roll)
        c = torch.cos(self.e)
        s = torch.sin(self.e)
        c1 = c[...,0]
        c2 = c[...,1]
        s1 = s[...,0]
        s2 = s[...,1]
        r11 = c1
        r12 = s1*s2
        r13 = c2*s1
        r21 = 0
        r22 = c2
        r23 = -s2
        r31 = -s1
        r32 = c1*s2
        r33 = c1*c2
        rot = torch.FloatTensor( [[r11, r12, r13],[r21, r22, r23],[r31, r32, r33]]  )
        return rot
