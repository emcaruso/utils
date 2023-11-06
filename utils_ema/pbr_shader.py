from utils_ema.geometry_SG import SphericalGaussians
from utils_ema.pbr_material import PBR_Material
from utils_ema.plot import plotter
import math
import torch
from tqdm import tqdm

TINY_NUMBER = 0.000001

class PBR_Shader():


    @staticmethod
    def render_spherical(shape, pixels, position, normal, view_dirs, material, light, device='cpu'):
        assert(isinstance(light,SphericalGaussians))
        assert(isinstance(material,PBR_Material))

        img = torch.zeros( shape , device=device)

        r = material.roughness
        k = ((r+1)**2)*0.125
        r4 = r**4
        r4_inv = 1/r4
        r4_inv_pi = r4_inv/math.pi
        m = material.metallic
        da = material.diffuse_albedo
        sa = material.specular_albedo



        # shape [ n_pixs, n_lights, R3 ]
        w_i = torch.nn.functional.normalize(light.lobe_axis, dim=-1).unsqueeze(0) # incident light
        w_o = - torch.nn.functional.normalize(view_dirs, dim=-1).unsqueeze(1)  # view direction
        n = torch.nn.functional.normalize(normal, dim=-1).unsqueeze(1)
        h = torch.nn.functional.normalize(w_i + w_o, dim=-1)
        # # get list of gaussian idxs in the emishperes for each view
        # gauss_mask = light.get_gauss_mask_hemisphere(n)

        # dot products
        wo_dot_h = torch.clamp(torch.sum(w_o*h, dim=-1), min=0)
        wo_dot_n = torch.clamp(torch.sum(w_o*n, dim=-1), min=0)
        wi_dot_n = torch.clamp(torch.sum(w_i*n, dim=-1), min=0)
        # mask = (wi_dot_n>0).unsqueeze(-1)

        # Light
        L_i = light

        # cosine gauss
        cos_sharp = torch.FloatTensor([0.0315]).unsqueeze(0).unsqueeze(0)
        cos_ampl = torch.FloatTensor([32.7080]).unsqueeze(0).unsqueeze(0)
        cos_dict = { 'lobe_axis':n, 'lobe_sharp': cos_sharp, 'lobe_ampl': cos_ampl }
        cos_gauss = SphericalGaussians( sgs_dict=cos_dict, device = position.device )
        cos_offs = 31.7003

        ####################
        ##### SPECULAR #####
        ####################

        # M
        fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h.unsqueeze(-1)) )
        geometric = ((wo_dot_n/(wo_dot_n*(1-k)+k)) * (wi_dot_n/(wi_dot_n*(1-k)+k))).unsqueeze(-1)
        M_D = torch.clamp(4*wo_dot_n*wi_dot_n, min=TINY_NUMBER).unsqueeze(-1)
        # M_N = fresnel # fresnel only
        # M_N = geometric # geometric only
        M_N = fresnel*geometric # complete
        M = M_N/M_D

        # brdf specular
        brdf_ampl = M*r4_inv_pi # complete
        # brdf_ampl = torch.FloatTensor([r4_inv_pi]).unsqueeze(0).unsqueeze(0) # no M
        brdf_sharp = torch.FloatTensor([2*r4_inv]).unsqueeze(0).unsqueeze(0).to(n.device)
        # brdf_dict = { 'lobe_axis': n, 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
        # Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = position.device )

        # warp brdf specular
        # brdf_axis = torch.nn.functional.normalize(2 * wi_dot_n.unsqueeze(-1) * n - w_i, dim=-1)
        brdf_axis = torch.nn.functional.normalize(2 * wo_dot_n.unsqueeze(-1) * n - w_o, dim=-1)
        # brdf_axis = 2 * wo_dot_n.unsqueeze(-1) * n - w_o
        brdf_sharp = brdf_sharp / (4 * torch.clamp(torch.sum(brdf_axis*n, dim=-1), min=0).unsqueeze(-1) + TINY_NUMBER)
        brdf_ampl = brdf_ampl
        brdf_dict = { 'lobe_axis': brdf_axis, 'lobe_sharp': brdf_sharp, 'lobe_ampl': brdf_ampl }
        Brdf_spec = SphericalGaussians( sgs_dict=brdf_dict, device = position.device )

        # integral
        # out = L_i*Brdf_spec
        # rgb_spec = torch.sum( spec_gauss.eval(w_i) , dim=1)
        # rgb_spec = torch.sum( (L_i*Brdf_spec*cos_gauss).eval(w_i)*mask , dim=1) - torch.sum( (L_i*Brdf_spec).eval(w_i)*cos_offs*mask , dim=1)
        # rgb_spec = torch.sum( (out).eval(w_i)*mask , dim=1)
        # rgb_spec = torch.sum( (L_i*Brdf_spec).eval(w_i)*mask , dim=1)  # no-lamb
        # rgb_spec = torch.sum( spec_gauss.eval(h) , dim=1)

        rgb_spec = (L_i*Brdf_spec*cos_gauss).hemisphere_int(n) - cos_offs*(L_i*Brdf_spec).hemisphere_int(n)
        rgb_spec = torch.sum(rgb_spec, dim=-2)
        # print(rgb_spec)
        # print(torch.max(rgb_spec))
        # print(torch.min(rgb_spec))
        # exit(1)

        ###################
        ##### DIFFUSE #####
        ###################

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

        # set pixels
        img[pixels[:,1],pixels[:,0]] = rgb
        
        return img


    # @staticmethod
    # def render_spherical(shape, pixels, position, normal, view_dirs, material, light, device='cpu'):
    #     assert(isinstance(light,SphericalGaussians))
    #     assert(isinstance(material,PBR_Material))

    #     img = torch.zeros( shape , device=device)

    #     r = material.roughness
    #     r4 = r**4
    #     m = material.metallic
    #     da = material.diffuse_albedo
    #     sa = material.specular_albedo


    #     # # get list of gaussian idxs in the emishperes for each view
    #     # gauss_mask = light.get_gauss_mask_hemisphere(normal)

    #     # shape [ n_pixs, n_lights, R3 ]
    #     w_i = torch.nn.functional.normalize(light.lobe_axis, dim=-1).unsqueeze(0) # incident light
    #     w_o = - torch.nn.functional.normalize(view_dirs, dim=-1).unsqueeze(1)  # view direction
    #     p = torch.nn.functional.normalize(position, dim=-1).unsqueeze(1)
    #     n = torch.nn.functional.normalize(normal, dim=-1).unsqueeze(1)

    #     h = torch.nn.functional.normalize(w_i + w_o, dim=-1)

    #     # dot products
    #     wo_dot_h = torch.clamp(torch.sum(w_o*h, dim=-1), min=0)
    #     wo_dot_n = torch.clamp(torch.sum(w_o*n, dim=-1), min=0)
    #     wi_dot_n = torch.clamp(torch.sum(w_i*n, dim=-1),min=0)

    #     # brdf specular terms
    #     k = ((r+1)**2)*0.125
    #     fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h.unsqueeze(-1)) )
    #     geometric = ((wo_dot_n/(wo_dot_n*(1-k)+k)) * (wi_dot_n/(wi_dot_n*(1-k)+k))).unsqueeze(-1)

    #     pr = 1/(math.pi*r4)
    #     normal_dict = { 'lobe_axis':n, 'lobe_sharp':torch.FloatTensor([2/r4]), 'lobe_ampl':torch.FloatTensor([pr,pr,pr]) }
    #     normal_distr = SphericalGaussians( sgs_dict=normal_dict, device = position.device )

    #     M_D = torch.clamp(4*wo_dot_n*wi_dot_n, min=0.00001).unsqueeze(-1)

    #     # M_N = fresnel # fresnel only
    #     # M_N = geometric # geometric only
    #     M_N = fresnel*geometric # complete

    #     M = M_N/M_D
    #     D = normal_distr.eval(h)
    #     # D = normal_distr.eval(h, gauss_mask)

    #     # f_s = D # specular brdf - only D
    #     # f_s = M # specular brdf - only M
    #     f_s = M*D # specular brdf - complete


    #     # brdf
    #     L_i = light.lobe_ampl
    #     # f_r = (da.unsqueeze(0).unsqueeze(0)/math.pi) # diffuse only
    #     # f_r = f_s # specular only
    #     f_r = (da.unsqueeze(0).unsqueeze(0)/math.pi)+f_s # complete

    #     # render
    #     # view_color = torch.sum(L_i.unsqueeze(0)*f_r, dim=1) # no lambertian
    #     view_color = torch.sum( (L_i*f_r)*wi_dot_n.unsqueeze(-1), dim=1) # lambertian

    #     img[pixels[:,1],pixels[:,0]] = view_color
        
    #     return img

