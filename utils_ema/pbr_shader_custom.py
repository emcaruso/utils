# from utils_ema.geometry_SG import SphericalGaussians
# from utils_ema.pbr_material import PBR_Material
# from utils_ema.plot import plotter
import gc
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
    def render_spherical(cls, cam, obj, scene, device='cpu'):

        with torch.no_grad():
            rgb, pixels = cls.render_spherical_pixels(cam, obj, scene)

            # set pixels
            res = (cam.intr.resolution[1],cam.intr.resolution[0],3)
            img = torch.zeros( res ).to(pixels.device)
            img[pixels[:,1],pixels[:,0],:] = rgb.to(img.device)

            return Image(img=img, device=device)

    @classmethod
    def render_spherical_loss(cls, cam, obj, scene, rgb_gt, device='cpu'):

        with torch.no_grad():
            rgb, pixels = cls.render_spherical_pixels(cam, obj, scene)

            # set pixels
            res = (cam.intr.resolution[1],cam.intr.resolution[0],3)

            img = torch.zeros( res ).to(pixels.device)
            img[pixels[:,1],pixels[:,0],:] = torch.abs(rgb.to(img.device) - rgb_gt[pixels[:,1],pixels[:,0],:])


        return Image(img=img, device=device)


    @staticmethod
    def render_spherical_pixels(cam, obj, scene, shading_percentage=1):

        gbuffers, pixs_all, _ = Renderer.get_buffers_pixels_dirs(cam, obj, shading_percentage=shading_percentage,channels=['mask','normal','uv'])
        # gbuffers, pixs_all, _ = Renderer.get_buffers_pixels_dirs(cam, obj, shading_percentage=1,channels=['mask','normal','uv'])

        
        if pixs_all is None: return None, None

        device = pixs_all.device

        pixs_all = pixs_all.to('cpu')
        normal_all = gbuffers["normal"].to('cpu')
        uv = gbuffers["uv"].to('cpu')
        mask_all = (gbuffers["mask"]==1).squeeze().to('cpu')

        # Image(torch.cat( (uv, uv), dim=-1)[...,:3]).show("uv",wk=0)

        segmentation_text = torch.flip( obj.textures[0].float(), [0] )
        s = torch.tensor(segmentation_text.shape)
        p = (uv*s[:2]).int()
        p = p[pixs_all[:,1],pixs_all[:,0],:]
        segm_pixs_all = (segmentation_text==1)[ p[:,1], p[:,0], : ]
        # Image(segmentation_text[...,0]).show(resolution_drop=4)
        # Image(segmentation_text[...,1]).show(resolution_drop=4)
        # Image(segmentation_text[...,2]).show(resolution_drop=4)
        # exit(1)

        segm_pixs = [ (s.squeeze()>0) for s in list(torch.split(segm_pixs_all, 1, -1)) ]

        # output image
        img = torch.zeros( tuple(list(mask_all.shape)+[3]) , dtype=torch.float32).to(device)


        pixels_list = []
        rgbs = []

        # for i in range(0,len(scene.materials_segm)):
        # for i in range(0,1):
        # for i in range(1,2):
        # for i in range(2,3):
        for i in range(1,3):

            light = scene.lights_segm[i]

            material = scene.materials_segm[i]

            # light.lobe_ampl = torch.clamp(light.lobe_ampl.clone(), min=0)

            # material.roughness = torch.clamp(material.roughness.clone(), min=0)
            # material.diffuse_albedo = torch.clamp(material.diffuse_albedo.clone(), min=0)
            # material.specular_albedo = torch.clamp(material.specular_albedo.clone(), min=0)
            # light.lobe_ampl = torch.pow(light.lobe_ampl.clone(), 2)
            # light.lobe_sharp = torch.pow(light.lobe_sharp.clone(), 2)
            # material.roughness = torch.abs(material.roughness.clone())
            # material.diffuse_albedo = torch.abs(material.diffuse_albedo.clone())
            # material.specular_albedo = torch.abs(material.specular_albedo.clone())


            mask = torch.zeros_like(mask_all)
            mask[ pixs_all[:,1], pixs_all[:,0] ] = segm_pixs[i]
            
            # Image(img=mask.type(torch.float32)).show(str(i),wk=0)

            pixs = cam.sample_rand_pixs_in_mask( mask, percentage=1)
            if pixs is None:
                continue
            pixels_list.append( pixs.clone() )

            normal = normal_all[pixs[:, 1], pixs[:, 0], :].to(device)
            view_dirs = cam.pix2dir( pixs ).to(device)

            # scene.plot_cams(1)
            # plotter.plot_ray(cam.pose.location().detach().cpu(), view_dirs.detach().cpu(), length=0.5, color='blue')
            # # plotter.plot_ray(position_val.detach().cpu(), normal_val.detach().cpu(), length=0.1, color='cyan')
            # plotter.wandb_log()

            r = material.roughness
            k = ((r+1)**2)*0.125
            r4 = r**4
            r4_inv = 1/(r4+1e-6)
            r4_inv_pi = r4_inv/math.pi
            # m = material.metallic
            da = material.diffuse_albedo
            sa = material.specular_albedo


            # l_sh = torch.pow(light.lobe_sharp, 2)
            l_sh = torch.exp(light.lobe_sharp)
            # l_am = torch.pow(light.lobe_ampl, 2).unsqueeze(0)
            l_am = torch.exp(light.lobe_ampl).unsqueeze(0)
            sgs_dict = { "lobe_axis":light.lobe_axis, "lobe_sharp": l_sh, "lobe_ampl": l_am }
            # sgs_dict = { "lobe_axis":light.lobe_axis, "lobe_sharp":light.lobe_sharp, "lobe_ampl":light.lobe_ampl }
            L_i = SphericalGaussians( sgs_dict = sgs_dict )
            # L_i = light
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

            rgb = torch.clamp(rgb, min=0, max=1.)

            rgbs.append(rgb)

        return torch.cat( rgbs, dim=0 ), torch.cat( pixels_list, dim=0 )


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
    def show_pbr_interactive(cls, camera, obj, scene, distance):
        with torch.no_grad():
            mover = MoverOrbital()
            User.detect_key()

            cam_copy = camera.clone(same_intr = True)

            obj.pose.set_euler(eul(e=torch.zeros([3]), device=obj.device))
            obj.pose.set_location(torch.zeros([3], device=obj.device) )

            while True:
                # update camera position
                pose = mover.get_pose(distance=distance).to(camera.device)
                cam_copy.pose.set_rotation(pose.rotation())
                cam_copy.pose.set_location(pose.location())

                image = cls.render_spherical(cam_copy, obj, scene)
                image.show(img_name="Pred",wk=1, resolution_drop=1)

                if 'q' in User.keys:
                    del mover
                    del cam_copy
                    break

        return False
