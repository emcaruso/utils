# from utils_ema.geometry_SG import SphericalGaussians
# from utils_ema.pbr_material import PBR_Material
# from utils_ema.plot import plotter
import time
import math
import cv2
import torch
from tqdm import tqdm
try:
    from .geometry_SG import SphericalGaussians
    from .geometry_direction import Direction
    from .geometry_euler import eul
    from .user_mover import MoverOrbital
    from .image import Image
    from .pbr_material import PBR_Material
    from .plot import plotter
    from .diff_renderer import Renderer
    from .user_interface import User
except:
    from geometry_SG import SphericalGaussians
    from geometry_direction import Direction
    from user_mover import MoverOrbitgal
    from image import Image
    from pbr_material import PBR_Material
    from plot import plotter
    from diff_renderer import Renderer
    from user_interface import User

TINY_NUMBER = 1e-5

class PBR_Shader():


    @classmethod
    def render_spherical(cls, cam, obj, light, device='cpu'):

        with torch.no_grad():
            rgb, pixels = cls.render_spherical_pixels(cam, obj, light)

            # set pixels
            res = (cam.intr.resolution[1],cam.intr.resolution[0],3)
            img = torch.zeros( res ).to(pixels.device)
            img[pixels[:,1],pixels[:,0],:] = rgb

            return Image(img=img, device=device)

    @staticmethod
    def render_spherical_pixels(cam, obj, L_i, shading_percentage=1):
        material = obj.material
        # assert(isinstance(light,SphericalGaussians))
        assert(isinstance(material,PBR_Material))

        # clamp light
        # L_i.lobe_sharp = torch.clamp_min(L_i.lobe_sharp, min=0.)
        # L_i.lobe_ampl = torch.clamp_min(L_i.lobe_ampl, min=0.)

        gbuffers, pixs, view_dirs = Renderer.get_buffers_pixels_dirs(cam, obj, shading_percentage=shading_percentage,channels=['mask','normal'])

        if pixs is None: return None, None

        # position = gbuffers["position"]
        normal = gbuffers["normal"]
        # normal = torch.nn.functional.normalize(normal, dim=-1)

        # cv2.imshow("mask",gbuffers['mask'].detach().cpu().numpy())
        # cv2.imshow("pos",position.detach().cpu().numpy())
        # cv2.imshow("n",normal.detach().cpu().numpy())
        # cv2.waitKey(1)

        # shape = normal.shape
        # position = position[pixs[:, 1], pixs[:, 0], :]
        normal = normal[pixs[:, 1], pixs[:, 0], :]
        # assert(torch.all(torch.eq(torch.norm(normal, dim=-1), 1)))

        # plotter.plot_points(position,color=color)

        r = material.roughness
        k = ((r+1)**2)*0.125
        r4 = r**4
        r4_inv = 1/(r4+1e-6)
        r4_inv_pi = r4_inv/math.pi
        # m = material.metallic
        da = material.diffuse_albedo
        sa = material.specular_albedo


        # shape [ n_pixs, n_lights, R3 ]
        w_i = torch.nn.functional.normalize(L_i.lobe_axis.vec3D(), dim=-1).unsqueeze(0) # incident light
        w_o = - torch.nn.functional.normalize(view_dirs, dim=-1).unsqueeze(1)  # view direction
        n = torch.nn.functional.normalize(normal, dim=-1).unsqueeze(1)
        h = torch.nn.functional.normalize(w_i + w_o, dim=-1)
        # # get list of gaussian idxs in the emishperes for each view
        # gauss_mask = light.get_gauss_mask_hemisphere(n)

        # dot products
        wo_dot_h = torch.clamp(torch.sum(w_o*h, dim=-1), min=TINY_NUMBER)
        wo_dot_n = torch.clamp(torch.sum(w_o*n, dim=-1), min=TINY_NUMBER)
        wi_dot_n = torch.clamp(torch.sum(w_i*n, dim=-1), min=TINY_NUMBER)
        # wo_dot_h = torch.sum(w_o*h, dim=-1)
        # wo_dot_n = torch.sum(w_o*n, dim=-1)
        # wi_dot_n = torch.sum(w_i*n, dim=-1)
        # mask = (wi_dot_n>0).unsqueeze(-1)

        # cosine gauss
        cos_sharp = torch.tensor([0.0315], device=n.device).unsqueeze(0).unsqueeze(0)
        cos_ampl = torch.tensor([32.7080], device=n.device).unsqueeze(0).unsqueeze(0)
        cos_dict = { 'lobe_axis':Direction(n), 'lobe_sharp': cos_sharp, 'lobe_ampl': cos_ampl }
        cos_gauss = SphericalGaussians( sgs_dict=cos_dict, device = normal.device )
        cos_offs = 31.7003

        ####################
        ##### SPECULAR #####
        ####################

        # M
        fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h.unsqueeze(-1)) )
        geometric = ((wo_dot_n/(wo_dot_n*(1-k)+k+1e-6)) * (wi_dot_n/(wi_dot_n*(1-k)+k+1e-6))).unsqueeze(-1)
        M_D = torch.clamp(4*wo_dot_n*wi_dot_n, min=TINY_NUMBER).unsqueeze(-1)
        # M_D = (4*wo_dot_n*wi_dot_n).unsqueeze(-1)
        M_N = fresnel*geometric
        M = M_N/M_D


        # brdf specular
        brdf_ampl = M*r4_inv_pi # complete
        # brdf_ampl = torch.FloatTensor([r4_inv_pi]).unsqueeze(0).unsqueeze(0) # no M
        brdf_sharp = torch.FloatTensor([2*r4_inv]).unsqueeze(0).unsqueeze(0).to(n.device)

        # brdf_dict = { 'lobe_axis': n, 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
        # Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = normal.device )

        # warp brdf specular
        # brdf_axis = torch.nn.functional.normalize(2 * wi_dot_n.unsqueeze(-1) * n - w_i, dim=-1)
        brdf_axis = torch.nn.functional.normalize(2 * wo_dot_n.unsqueeze(-1) * n - w_o, dim=-1)
        brdf_sharp = brdf_sharp / (4 * wo_dot_n.unsqueeze(-1) + TINY_NUMBER)
        # brdf_axis = 2 * wo_dot_n.unsqueeze(-1) * n - w_o
        # brdf_sharp = brdf_sharp / (4 * torch.sum(brdf_axis*n, dim=-1).unsqueeze(-1) + TINY_NUMBER)
        # brdf_sharp = brdf_sharp / (4 * torch.clamp(torch.sum(n, dim=-1), min=TINY_NUMBER).unsqueeze(-1) + TINY_NUMBER)
        brdf_ampl = brdf_ampl
        brdf_dict = { 'lobe_axis': Direction(brdf_axis), 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
        Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = normal.device )

        # integral
        # out = L_i*Brdf_spec
        # rgb_spec = torch.sum( spec_gauss.eval(w_i) , dim=1)
        # rgb_spec = torch.sum( (L_i*Brdf_spec*cos_gauss).eval(w_i)*mask , dim=1) - torch.sum( (L_i*Brdf_spec).eval(w_i)*cos_offs*mask , dim=1)
        # rgb_spec = torch.sum( (out).eval(w_i)*mask , dim=1)
        # rgb_spec = torch.sum( (L_i*Brdf_spec).eval(w_i)*mask , dim=1)  # no-lamb
        # rgb_spec = torch.sum( spec_gauss.eval(h) , dim=1)

        rgb_spec = (L_i*Brdf_spec*cos_gauss).hemisphere_int(n) - cos_offs*(L_i*Brdf_spec).hemisphere_int(n)
        # rgb_spec = (L_i*Brdf_spec*cos_gauss).hemisphere_int(n) - cos_offs*(L_i*Brdf_spec).hemisphere_int(n)
        # rgb_spec = (L_i*cos_gauss).hemisphere_int(n) - cos_offs*(L_i).hemisphere_int(n)
        rgb_spec = torch.sum(rgb_spec, dim=-2)

        ##################
        #### DIFFUSE #####
        ##################

        # # integral
        diff_term = (da.unsqueeze(0).unsqueeze(0)/math.pi)
        # # rgb_diff = torch.sum( diff_term , dim=1)                                                          #diff no-lamb no-light
        # # rgb_diff = torch.sum( L_i.eval(w_i)*diff_term , dim=1)                                      #diff no-lamb    light
        # # rgb_diff = torch.sum( cos_gauss.eval(w_i)*mask , dim=1) - cos_offs*L_i.n_sgs                        #cos only
        # # rgb_diff = torch.sum( cos_gauss.eval(w_i)*diff_term , dim=1) - cos_offs*diff_term*L_i.n_sgs    #diff    lamb no-light
        # rgb_diff = torch.sum( (L_i*cos_gauss).eval(w_i)*diff_term*mask , dim=1) - torch.sum( L_i.eval(w_i)*diff_term*cos_offs*mask , dim=1)  #diff    lamb    light
        rgb_diff = ( (L_i*cos_gauss).hemisphere_int(n) - cos_offs*(L_i).hemisphere_int(n) )*diff_term
        rgb_diff = torch.sum(rgb_diff, dim=-2)

        # rgb = rgb_diff
        # rgb = rgb_spec
        rgb = rgb_diff+rgb_spec

        rgb = torch.clamp(rgb, min=TINY_NUMBER, max=1.)

        return rgb, pixs


    #@staticmethod
    #def render_spherical_pixels(cam, obj, light):
    #    material = obj.material
    #    assert(isinstance(light,SphericalGaussians))
    #    assert(isinstance(material,PBR_Material))


    #    gbuffers, pixs, view_dirs = Renderer.get_buffers_pixels_dirs(cam, obj, shading_percentage=1)
    #    position = gbuffers["position"]
    #    normal = gbuffers["normal"]

    #    # cv2.imshow("pos",position.detach().cpu().numpy())
#    # cv2.imshow("n",normal.detach().cpu().numpy())
    #    # cv2.waitKey(0)

    #    shape = normal.shape
    #    position = position[pixs[:, 1], pixs[:, 0], :]
    #    normal = normal[pixs[:, 1], pixs[:, 0], :]

    #    # plotter.plot_points(position,color=color)

    #    r = material.roughness
    #    k = ((r+1)**2)*0.125
    #    r4 = r**4
    #    r4_inv = 1/r4
    #    r4_inv_pi = r4_inv/math.pi m = material.metallic da = material.diffuse_albedo sa = material.specular_albedo

    #    # shape [ n_pixs, n_lights, R3 ]
    #    w_i = torch.nn.functional.normalize(light.lobe_axis.vec3D(), dim=-1).unsqueeze(0) # incident light
    #    w_o = - torch.nn.functional.normalize(view_dirs, dim=-1).unsqueeze(1)  # view direction
    #    n = torch.nn.functional.normalize(normal, dim=-1).unsqueeze(1)
    #    h = torch.nn.functional.normalize(w_i + w_o, dim=-1)
    #    # # get list of gaussian idxs in the emishperes for each view
    #    # gauss_mask = light.get_gauss_mask_hemisphere(n)

    #    # dot products
    #    wo_dot_h = torch.clamp(torch.sum(w_o*h, dim=-1), min=0)
    #    wo_dot_n = torch.clamp(torch.sum(w_o*n, dim=-1), min=0)
    #    wi_dot_n = torch.clamp(torch.sum(w_i*n, dim=-1), min=0)
    #    # mask = (wi_dot_n>0).unsqueeze(-1)

    #    # Light
    #    L_i = light

    #    # cosine gauss
    #    cos_sharp = torch.tensor([0.0315], device=n.device).unsqueeze(0).unsqueeze(0)
    #    cos_ampl = torch.tensor([32.7080], device=n.device).unsqueeze(0).unsqueeze(0)
    #    cos_dict = { 'lobe_axis':Direction(n), 'lobe_sharp': cos_sharp, 'lobe_ampl': cos_ampl }
    #    cos_gauss = SphericalGaussians( sgs_dict=cos_dict, device = position.device )
    #    cos_offs = 31.7003

    #    ####################
    #    ##### SPECULAR #####
    #    ####################

    #    # M
    #    fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h.unsqueeze(-1)) )
    #    geometric = ((wo_dot_n/(wo_dot_n*(1-k)+k)) * (wi_dot_n/(wi_dot_n*(1-k)+k))).unsqueeze(-1)
    #    M = (fresnel*geometric)/(torch.clamp(4*wo_dot_n*wi_dot_n, min=TINY_NUMBER).unsqueeze(-1))

    #    # brdf specular
    #    brdf_ampl = M*r4_inv_pi # complete
    #    # brdf_ampl = torch.FloatTensor([r4_inv_pi]).unsqueeze(0).unsqueeze(0) # no M
    #    brdf_sharp = torch.FloatTensor([2*r4_inv]).unsqueeze(0).unsqueeze(0).to(n.device)
    #    # brdf_dict = { 'lobe_axis': n, 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
    #    # Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = position.device )

    #    # warp brdf specular
    #    # brdf_axis = torch.nn.functional.normalize(2 * wi_dot_n.unsqueeze(-1) * n - w_i, dim=-1)
    #    brdf_axis = torch.nn.functional.normalize(2 * wo_dot_n.unsqueeze(-1) * n - w_o, dim=-1)
    #    # brdf_axis = 2 * wo_dot_n.unsqueeze(-1) * n - w_o
    #    brdf_sharp = brdf_sharp / (4 * torch.clamp(torch.sum(brdf_axis*n, dim=-1), min=0).unsqueeze(-1) + TINY_NUMBER)
    #    brdf_ampl = brdf_ampl
    #    brdf_dict = { 'lobe_axis': Direction(brdf_axis), 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
    #    Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = position.device )

    #    rgb_spec = (L_i*Brdf_spec*cos_gauss).hemisphere_int(n) - cos_offs*(L_i*Brdf_spec).hemisphere_int(n)
    #    # Li_brdf = L_i*Brdf_spec
    #    # Li_brdf_cos = L_i*Brdf_spec*cos_gauss

    #    # s1 = (Li_brdf_cos).hemisphere_int(n)
    #    # s2 = cos_offs*(Li_brdf).hemisphere_int(n)
    #    # s1 = (L_i*Brdf_spec*cos_gauss).hemisphere_int(n)
    #    # s2 = cos_offs*(L_i*Brdf_spec).hemisphere_int(n)
    #    # rgb_spec = s1 - s2
    #    # rgb_spec = s1
    #    rgb_spec = torch.sum(rgb_spec, dim=-2)

    #    ###################
    #    ##### DIFFUSE #####
    #    ###################

    #    # # integral
    #    diff_term = (da.unsqueeze(0).unsqueeze(0)/math.pi)
    #    # # rgb_diff = torch.sum( diff_term , dim=1)                                                          #diff no-lamb no-light
    #    # # rgb_diff = torch.sum( L_i.eval(w_i)*diff_term , dim=1)                                      #diff no-lamb    light
    #    # # rgb_diff = torch.sum( cos_gauss.eval(w_i)*mask , dim=1) - cos_offs*L_i.n_sgs                        #cos only
    #    # # rgb_diff = torch.sum( cos_gauss.eval(w_i)*diff_term , dim=1) - cos_offs*diff_term*L_i.n_sgs    #diff    lamb no-light
    #    # rgb_diff = torch.sum( (L_i*cos_gauss).eval(w_i)*diff_term*mask , dim=1) - torch.sum( L_i.eval(w_i)*diff_term*cos_offs*mask , dim=1)  #diff    lamb    light
    #    rgb_diff = ( (L_i*cos_gauss).hemisphere_int(n) - cos_offs*(L_i).hemisphere_int(n) )*diff_term
    #    rgb_diff = torch.sum(rgb_diff, dim=-2)

    #    ###################

    #    # rgb = rgb_diff
    #    # rgb = rgb_spec
    #    rgb = rgb_diff+rgb_spec

        
    #    return rgb, pixs


    @classmethod
    def show_pbr_interactive(cls, camera, obj, light, distance):
        with torch.no_grad():
            mover = MoverOrbital()
            User.detect_key()

            cam_copy = camera.clone(same_intr = True)

            obj.pose.set_euler(eul(e=torch.zeros([3]), device=obj.device))
            obj.pose.set_location(torch.zeros([3], device=obj.device) )

            while True:
                # update camera position
                pose = mover.get_pose(distance=distance).to(camera.device)
                # plotter.plot_cam(cam_copy)
                # plotter.plot_object(obj)
                # cam_copy.pose = pose
                cam_copy.pose.set_rotation(pose.rotation())
                cam_copy.pose.set_location(pose.location())
                # plotter.plot_cam(cam_copy)
                # plotter.show()
                image = cls.render_spherical(cam_copy, obj, light)
                image.show(img_name="Pred",wk=1, resolution_drop=1)

                if 'q' in User.keys:
                    del mover
                    del cam_copy
                    break

        return False
