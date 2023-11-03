from utils_ema.geometry_SG import SphericalGaussians
from utils_ema.pbr_material import PBR_Material
from utils_ema.plot import plotter
import math
import torch
from tqdm import tqdm

class PBR_Shader():

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


    #     # get list of gaussian idxs in the emishperes for each view
    #     gauss_list =light.get_list_of_gauss_for_each_view(normal)
    #     # get_gauss_mask_hemisphere

    #     results = []
    #     # for each view
    #     for view_idx, gauss_idxs in tqdm(enumerate(gauss_list)):
    #         w_i = light.lobe_axis[gauss_idxs] # incident light
    #         w_o = view_dirs[view_idx]  # view direction
    #         p = position[view_idx]
    #         n = normal[view_idx]


    #         # compute half-way vector
    #         h = torch.nn.functional.normalize(w_i+w_o,dim=-1)

    #         # dot products
    #         wo_dot_h = torch.sum(w_o*h, dim=-1)
    #         wo_dot_n = torch.sum(w_o*n, dim=-1)
    #         wi_dot_n = torch.sum(w_i*n, dim=-1)

    #         # brdf specular terms
    #         k = (r+1)**2*0.125
    #         k_term = (1-k)+k
    #         fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h .unsqueeze(-1)) )
    #         geometric = (wo_dot_n/(wo_dot_n*k_term)) * (wi_dot_n/(wi_dot_n*k_term))

    #         pr = 1/(math.pi*r4)
    #         normal_dict = { 'lobe_axis':n.unsqueeze(0), 'lobe_sharp':torch.FloatTensor([2/r4]), 'lobe_ampl':torch.FloatTensor([pr,pr,pr]) }
    #         normal_distr = SphericalGaussians( sgs_dict=normal_dict, device = position.device )

    #         M_D = (4*wo_dot_n*wi_dot_n).unsqueeze(-1)
    #         M_N = fresnel*geometric.unsqueeze(-1)
    #         M = M_N/M_D
    #         D = normal_distr.eval(h)

    #         f_s = M*D # specular brdf

    #         # brdf
    #         L_i = light.lobe_ampl[gauss_idxs]
    #         f_r = (da/math.pi)+f_s

    #         # render
    #         view_color = torch.sum(L_i*f_r*wi_dot_n.unsqueeze(-1), dim=0)
    #         img[pixels[view_idx,0],pixels[view_idx,1]] = view_color
    #     return img

    @staticmethod
    def render_spherical(shape, pixels, position, normal, view_dirs, material, light, device='cpu'):
        assert(isinstance(light,SphericalGaussians))
        assert(isinstance(material,PBR_Material))

        img = torch.zeros( shape , device=device)

        r = material.roughness
        r4 = r**4
        m = material.metallic
        da = material.diffuse_albedo
        sa = material.specular_albedo


        # get list of gaussian idxs in the emishperes for each view
        gauss_mask = light.get_gauss_mask_hemisphere(normal)

        w_i = torch.nn.functional.normalize(light.lobe_axis, dim=-1) # incident light
        w_o = torch.nn.functional.normalize(view_dirs, dim=-1)  # view direction
        p = torch.nn.functional.normalize(position, dim=-1)
        n = torch.nn.functional.normalize(normal, dim=-1)

        h = torch.nn.functional.normalize(w_i.unsqueeze(0) + w_o.unsqueeze(1), dim=-1)
        h_flat = h[gauss_mask>0]

        # dot products
        wo_dot_h = torch.sum(w_o.unsqueeze(1)*h, dim=-1)
        wo_dot_n = torch.sum(w_o*n, dim=-1).unsqueeze(1)
        # wi_dot_n = torch.sum(w_i.unsqueeze(0)*n.unsqueeze(1), dim=-1)
        wi_dot_n = torch.clamp(torch.sum(w_i.unsqueeze(0)*n.unsqueeze(1), dim=-1),min=0)
        # torch.clamp(input, min=None, max=None, *, out=None) 

        # # test wi_dot_n
        # print( torch.min(wi_dot_n))
        # print(wi_dot_n.shape)
        # exit(1)


        # brdf specular terms
        k = (r+1)**2*0.125
        fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h .unsqueeze(-1)) )
        geometric = (wo_dot_n/(wo_dot_n*(1-k)+k)) * (wi_dot_n/(wi_dot_n*(1-k)+k))

        pr = 1/(math.pi*r4)
        normal_dict = { 'lobe_axis':n, 'lobe_sharp':torch.FloatTensor([2/r4]), 'lobe_ampl':torch.FloatTensor([pr,pr,pr]) }
        normal_distr = SphericalGaussians( sgs_dict=normal_dict, device = position.device )

        M_D = (4*wo_dot_n*wi_dot_n).unsqueeze(-1)
        M_N = fresnel*geometric.unsqueeze(-1)
        M = M_N/M_D
        D = normal_distr.eval_batch(h, gauss_mask)

        f_s = torch.mean(D, dim=-1) # specular brdf
        # print(f_s)

        # brdf
        L_i = light.lobe_ampl
        f_r = (da.unsqueeze(0).unsqueeze(0)/math.pi)+f_s.unsqueeze(-1)
        # f_r = (da.unsqueeze(0).unsqueeze(0)/math.pi)

        # print(torch.max(wi_dot_n))
        # print(torch.max(L_i.unsqueeze(0)*f_r))
        # print(torch.max(wi_dot_n))
        # print( L_i.shape)
        # print( f_r.shape)
        # print( wi_dot_n.shape)
        # print( torch.min(wi_dot_n))
        # print(f_r.shape)
        # exit(1)

        # render
        # view_color = torch.sum(L_i.unsqueeze(0)*f_r, dim=1)
        # view_color = torch.sum(L_i.unsqueeze(0)*f_r*wi_dot_n.unsqueeze(-1), dim=1)
        view_color = torch.sum( (L_i.unsqueeze(0)*f_r)*wi_dot_n.unsqueeze(-1), dim=1)
        # print(torch.max(view_color))
        # print(torch.min(view_color))
        img[pixels[:,1],pixels[:,0]] = view_color
        # print(torch.max(img))
        
        return img


        # i=0
        # # debug
        # print(w_i.shape)
        # print(w_i[gauss_mask[i,:]>0].shape)
        # print(w_o.shape)
        # print(p.shape)
        # print(n.shape)
        # print(h.shape)
        # plotter.plot_ray(p[i,:].view(1,3),w_i[:1,:], color='blue')
        # plotter.plot_ray(p[i,:].view(1,3),h[i,:1,:], length=1, color='magenta')
        # plotter.plot_ray(p[i,:].unsqueeze(0).repeat(w_i[gauss_mask[i,:]].shape[0],1),w_i[gauss_mask[i,:]>0], length=0.1, color='cyan') #hemisphere
        # plotter.plot_ray(p[i,:].unsqueeze(0),w_o[i,:].unsqueeze(0))
        # plotter.plot_ray(p[i,:].unsqueeze(0),n[i,:].unsqueeze(0), color='green')
        # plotter.show()
        # exit(1)

        # print(h.shape)
        # h = torch.nn.functional.normalize(h,dim=-1)
        # print(h.shape)
        # print(w_i.shape)
        # print(w_o.shape)
        # print(p.shape)
        # print(n.shape)
        # exit(1)

#         results = []
#         # for each view
#         for view_idx, gauss_idxs in tqdm(enumerate(gauss_list)):
#             w_i = light.lobe_axis[gauss_idxs] # incident light
#             w_o = view_dirs[view_idx]  # view direction
#             p = position[view_idx]
#             n = normal[view_idx]


#             # compute half-way vector
#             h = torch.nn.functional.normalize(w_i+w_o,dim=-1)

#             # dot products
#             wo_dot_h = torch.sum(w_o*h, dim=-1)
#             wo_dot_n = torch.sum(w_o*n, dim=-1)
#             wi_dot_n = torch.sum(w_i*n, dim=-1)

#             # brdf specular terms
#             k = (r+1)**2*0.125
#             k_term = (1-k)+k
#             fresnel = torch.pow( sa.unsqueeze(0) + (1-sa.unsqueeze(0))*2, ( (-5.55474*wo_dot_h.unsqueeze(-1)+6.8316)*wo_dot_h .unsqueeze(-1)) )
#             geometric = (wo_dot_n/(wo_dot_n*k_term)) * (wi_dot_n/(wi_dot_n*k_term))

#             pr = 1/(math.pi*r4)
#             normal_dict = { 'lobe_axis':n.unsqueeze(0), 'lobe_sharp':torch.FloatTensor([2/r4]), 'lobe_ampl':torch.FloatTensor([pr,pr,pr]) }
#             normal_distr = SphericalGaussians( sgs_dict=normal_dict, device = position.device )

#             M_D = (4*wo_dot_n*wi_dot_n).unsqueeze(-1)
#             M_N = fresnel*geometric.unsqueeze(-1)
#             M = M_N/M_D
#             D = normal_distr.eval(h)

#             f_s = M*D # specular brdf

#             # brdf
#             L_i = light.lobe_ampl[gauss_idxs]
#             f_r = (da/math.pi)+f_s

#             # render
#             view_color = torch.sum(L_i*f_r*wi_dot_n.unsqueeze(-1), dim=0)
#             img[pixels[view_idx,0],pixels[view_idx,1]] = view_color
#         return img

