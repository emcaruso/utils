from utils_ema.camera_cv import Camera_cv, Intrinsics
from utils_ema.geometry_SG import SphericalGaussians
from utils_ema.geometry_direction import Direction
from utils_ema.diff_renderer import Renderer
from utils_ema.objects import Object
from utils_ema.image import Image
from utils_ema.torch_utils import get_device
import trimesh
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
from utils_ema.plot import plotter

import numpy as np
from tqdm import tqdm


class ShadowMap():

    def __init__(self, n_lights, device='cpu', requires_grad=False):
        self.n_lights = n_lights
        self.device = device
        self.requires_grad = requires_grad

    def compute(self, obj):
        v = obj.mesh.vertices
        f = obj.mesh.indices
        d = Direction.sample_directions(self.n_lights, self.device, self.requires_grad).vec3D()
        m = trimesh.Trimesh(vertices=v, faces=f)

        shadow_map = torch.zeros( [len(v),len(d)] , dtype=torch.bool )

        for d_id, dir in enumerate(tqdm(d)):
            n_chunks = 100
            chunk_len = int(len(v)/n_chunks)
            ranges = [ ((i*chunk_len),((i+1)*chunk_len)) for i in range(n_chunks) ]
            for r in ranges:
                chunk_d = dir.unsqueeze(0).repeat(chunk_len, 1)
                chunk_v = v[ r[0]:r[1], ...]
                v_ids = list(range(r[0],r[1]))
                # print(i)
                # print(chunk_d.shape)
                # print(chunk_v.shape)
                _, index_ray, _ = m.ray.intersects_location(
                    ray_origins = chunk_v.cpu(),
                    ray_directions = chunk_d.cpu()
                )
                index_ray = torch.from_numpy(index_ray).to(self.device)
                _, counts = torch.unique(index_ray, return_counts=True)
                shadow_map[v_ids,d_id] = (counts>1).cpu()
                assert(torch.min(counts)!=0)
                # # print(torch.mcounts)
                # # print(counts)
                # # print(r[1])
        # print(len(v))
        # print(shadow_map[0,-700:])
        exit(1)



#             chunks = [ v[(i*n_chunks):(i*n_chunks+chunk_len), ...] for i in range(n_chunks) ]
#             # chunks = [ v[i:100, ...] for i in range(n_chunks) ]
#             for i, chunk in enumerate(chunks):

#                 chunk_d = dir.unsqueeze(0).repeat(chunk_len, 1)
#                 _, index_ray, _ = m.ray.intersects_location(
#                     ray_origins = chunk,
#                     ray_directions = chunk_d
#                 )
#                 # o = np.unique(index_ray)
#                 _, counts = np.unique(index_ray, return_counts=True)

#                 print(counts>1)

                # # print(counts.shape)
                # # print(counts)
                # # print(np.min(counts))
                # # print(np.unique(counts))
                # # print(np.min(o))
                # # print(o.shape)
                # # exit(1)
                # print(type(index_ray))
                # print(index_ray.shape)
                # print(index_ray)
                # print(chunk.shape)
                # # print(chunk_d.shape)
                # exit(1)
                


        # for vert in tqdm(v):
        #     # print(vert.shape)
        #     vert = vert.unsqueeze(0).repeat(len(d), 1)
        #     # print(vert.shape)
        #     # print(d.shape)
        #     # exit(1)
        #     locations, index_ray, index_tri = m.ray.intersects_location(
        #         ray_origins = vert,
        #         ray_directions = d
        #     )


        # print(v.shape)
        # exit(1)
        # ray_origins = np.tile(V, (len(S), 1))  # Repeat V for each direction in S
        # ray_directions = np.repeat(S, len(V), axis=0)  # Repeat each direction in S for each vertex in V
        # locations, index_ray, index_tri = mesh.ray.intersects_location(
        #     ray_origins=ray_origins,
        #     ray_directions=ray_directions
        # )

class customDataset():

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx,...], self.Y[idx]


class MLP(nn.Module):

    def __init__(self, sizes, activations):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(sizes[0], sizes[1])
        self.fc2 = nn.Linear(sizes[1], sizes[2])
        self.fc3 = nn.Linear(sizes[2], sizes[3])

        # self.fcs = []
        # self.act = []
        # act_dict = { "relu": F.relu , "sigmoid": F.sigmoid }
        # for i in range(len(sizes)-1):
        #     self.fcs.append(nn.Linear(sizes[i], sizes[i+1]))
        #     self.act.append(act_dict[activations[i]])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        # for i, fc in enumerate(self.fcs):
        #     x = fc(x)
        #     x = act_funcs[ self.act[i] ](x)
        return x
        
class ShadowMapNeural():

    def __init__(self, cfg, obj, device='cpu'):
        self.cfg = cfg
        self.device = device
        self.obj = obj
        self.cams = []

    def generate_data(self):
        with torch.no_grad():
            directions = Direction.sample_directions(self.cfg.cameras.n, device=self.device)
            intr = Intrinsics(K=torch.FloatTensor(self.cfg.cameras.K),
                              resolution=torch.LongTensor(self.cfg.cameras.resolution),
                              sensor_size=torch.FloatTensor(self.cfg.cameras.sensor_size))
            for i in range(len(directions.azel)):
                d = Direction(directions.azel[i,...])
                pose = d.to_pose_on_sphere(distance=self.cfg.cameras.radius)
                cam = Camera_cv(intr, pose, name="Sphere_"+str(i).zfill(3))
                self.cams.append(cam)
                # plotter.plot_cam(cam)
            # plotter.wandb_log()

            p_all = self.obj.mesh.vertices.to(self.device)
            n_all = self.obj.mesh.vertex_normals.detach().to(self.device)
            # plotter.plot_ray(p_all[:2000,...],n_all[:2000,...], length=0.02)
            # plotter.wandb_log()
            # exit(1)
            # print(p_all.shape)
            # exit(1)
            # X_list = torch.FloatTensor( size=[0,9] )
            X_list = []
            Y_list = []
            for cam in tqdm(self.cams):

                gbuffers, rast, idx = Renderer.diffrast(cam, self.obj, channels=["position", "normal", "mask"], get_rast_idx=True)
                rast = rast.squeeze(0)

                mask = (gbuffers["mask"] > 0).squeeze()
                pixs = cam.sample_rand_pixs_in_mask( mask, percentage=1)

                tri_ids = (rast[pixs[:,1], pixs[:,0]][...,-1]-1).to(torch.int)
                v_idx = torch.unique(idx[tri_ids].view(-1)).to(self.device)
                d_all = p_all-cam.pose.location().unsqueeze(0)
                # d_all = d_all/torch.norm(d_all)
                d_all = torch.nn.functional.normalize(d_all, dim=-1)

                X = torch.cat( (p_all, n_all, d_all ), dim=1 )
                Y = torch.zeros( [p_all.shape[0],1], dtype=torch.float32 )
                Y[v_idx] = 1

                # X_list = torch.cat( ( X_list, X ), dim=0 )
                X_list.append(X)
                Y_list.append(Y)

            X = torch.cat( X_list, dim=0 )
            Y = torch.cat( Y_list, dim=0 )
            self.dataset_train = customDataset(X, Y)

            # print("split")
            # X_train_val, X_test, Y_train_val, Y_test = train_test_split( X, Y, test_size=0.1, stratify=Y)
            # X_train, X_val, Y_train, Y_val = train_test_split( X_train_val, Y_train_val, test_size=0.2, stratify=Y)
            # self.dataset_train = customDataset(X_train, Y_train)
            # self.dataset_val = customDataset(X_val, Y_val)
            # self.dataset_test = customDataset(X_test, Y_test)
            

                # print(idx.shape)
                # print(idx)

                # # print(rast.shape)
                # # print(tri_ids.shape)
                # exit(1)
                # position = gbuffers["position"][pixs[:,1],pixs[:,0]]
                # normal = gbuffers["normal"][pixs[:,1],pixs[:,0]]
                # print(position.shape)
                # print(normal.shape)
                # exit(1)

                # gbuffers["position"][pixs[:,1],pixs[:,0]] = 1
                # Image(gbuffers["position"]).show(resolution_drop=4, wk=1)




    def train(self):
        train_dataloader = DataLoader(self.dataset_train, batch_size=64, shuffle=True)

        device = get_device()

        model = MLP( self.cfg.train.sizes, self.cfg.train.activations).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.train.lr)
        
        for epoch in range(self.cfg.train.n_epochs):
            model.train()
            loss_total = torch.FloatTensor([0]).to(device)
            c = 0
            for x,y in train_dataloader:
                optimizer.zero_grad()
                outputs = model(x.to(device))
                loss = criterion(outputs, y.to(device))
                loss.backward()
                optimizer.step()
                loss_total += loss.detach()
                c+=1
            loss_total/=c

            wandb.log( {"BCE loss: ": loss_total.item()} )
            print(loss_total)










        


