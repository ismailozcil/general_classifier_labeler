import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFont, ImageDraw, ImageEnhance, ImageFile
import cv2
# importing just because PIL refuses to read saved image
import  scipy.io
import gdown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.io import read_image

from ultralytics import YOLO

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
from IPython.display import Image as IpyImage




class ImageLabeler:
    """
    Labels objects in an image with the labels in the dataset.
    Multi Label to an image or classification of image can be done.

    This class combines object detection, feature extraction, and classification
    to identify objects in an image and predict their possible labels.

    Args:
        model_name (str, optional): Name of the feature extraction model ('resnet18' or 'regnet_y'). Defaults to 'resnet18'.
        dataset_name (str, optional): Name of the dataset used for learning. Defaults to 'RGBDAffordance'.
        img_size (int, optional): Size of the input image. Defaults to 224.
        device (str, optional): Device to run the model on ('cpu' or 'gpu'). Defaults to 'cpu'.
        nr_of_bases (int, optional): Number of bases for subspace projection. Defaults to 20.
        auto_threshold (float, optional): Automatic threshold for energy calculation. Defaults to 0.8.
        plot_graph (bool, optional): Whether to plot graphs during analysis. Defaults to False.

    Attributes:
        model_name (str): Name of the feature extraction model.
        dataset_name (str): Name of the dataset used for training.
        img_size (int): Size of the input image.
        device (torch.device): Device to run the model on.
        nr_of_bases (int): Number of bases for subspace projection.
        auto_threshold (float): Automatic threshold for energy calculation.
        plot_graph (bool): Whether to plot graphs during analysis.
        object_labels (list): List of object labels.
        affordance_names_T (list): List of affordance names in title case.
        affordance_names (list): List of affordance names in lowercase.
        model_name_T (str): Title case version of the model name.
        W_matr (torch.Tensor): Weight matrix for affordance classification.
        base_list (dict): Dictionary storing subspace bases for each affordance.
        base_point_vecs (dict): Dictionary storing base point vectors for each affordance.
        state_dict (dict): Dictionary storing state vectors for each affordance.
        threshold_dict (dict): Dictionary storing thresholds for each affordance.
        afford_labellist (list): List of affordance labels.
        afford_dict (dict): Dictionary mapping affordance labels to lowercase names.
        afford_dict_T (dict): Dictionary mapping affordance labels to title case names.
        softmax (torch.nn.Module): Softmax function for probability calculation.
    """
    def __init__(self, model_name='resnet18', dataset_name='RGBDAffordance', img_size=224, device= 'cpu', nr_of_bases=20, auto_threshold = 0.8, plot_graph=False):
        # Check if input parameters are valid
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name must be a string.")
        if not isinstance(img_size, int):
            raise TypeError("img_size must be an integer.")
        if not isinstance(device, str):
            raise TypeError("device must be a string.")
        if not isinstance(nr_of_bases, int):
            raise TypeError("nr_of_bases must be an integer.")
        if not isinstance(auto_threshold, float):
            raise TypeError("auto_threshold must be a float.")
        if not isinstance(plot_graph, bool):
            raise TypeError("plot_graph must be a boolean.")

    def __init__(self, model_name='resnet18', dataset_name='RGBDAffordance', img_size=224, device= 'cpu', nr_of_bases=20, auto_threshold = 0.8, plot_graph=False):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.img_size = img_size
        if device =='gpu':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        #self.device = device
        self.nr_of_bases = nr_of_bases
        self.auto_threshold = auto_threshold
        self.plot_graph = plot_graph
        
        self.model_name_T = 'ResNet18' if model_name == 'resnet18' else 'RegNetY'
        w_matr =np.loadtxt(r'/content/W_matr_%s.csv'%self.model_name, delimiter=',')
        w_max = np.max(w_matr)
        self.W_matr = torch.tensor(w_matr/w_max)

        self.base_list = dict()
        self.base_point_vecs = dict()
        self.state_dict = dict()
        self.threshold_dict = dict()
        self.afford_labellist = list()
        self.afford_dict = dict()
        self.afford_dict_T = dict()
        self.softmax = nn.Softmax()

    def get_tens(self):
        """Loads state tensors, result tensors, and image names from saved files.
    
        This method loads the state tensors, result tensors, and image names
        from previously saved files based on the model name, dataset name,
        and image size.
    
        Returns:
            tuple: A tuple containing the state tensors, result tensors,
                   and a list of image names.
        """
        # state vectors obtained from Resnet are loaded from previously saved file
        st_tns = torch.load(r'/content/%s_with_%s_%dnetworkout.pt'%(self.model_name, self.dataset_name, self.img_size))
        st_tns = torch.transpose(st_tns, 0, 1)
        # also results of those tensor are loaded. Results contain affordance labels, like 0, 1, 2,
        rslt_tns = torch.load(r'/content/%s_with_%s_%dlabel.pt'%(self.model_name, self.dataset_name, self.img_size))

        # image paths are loaded to name list
        # image names also contain names of the object/s in the image
        nm_list = []
        f = open(r'/content/%s_with_%s_%dimagepaths.txt'%(self.model_name, self.dataset_name, self.img_size), 'r')
        for line in f:
            nm_list= line.strip().split('#')
        f.close()

        return st_tns, rslt_tns, nm_list

    def project_matr(self, bases):
        """Calculates the projection matrix for a given set of bases.
        
        The projection matrix projects vectors onto the subspace spanned by the bases.
        
        Args:
          bases: A TALL tensor representing the bases of the subspace.
          
        Returns:
          A tensor representing the projection matrix.
        """
        # Calculate the projection matrix using the pseudoinverse formula for tall matrices: bases * (bases^T * bases)^-1 * bases^T
        # Actually, it can be calculated directly: bases * bases^T, but this formula is only for orthonormal bases.
        # This case is orthonormal, but to use the function in general cases, general formula is used even though it is more 
        # calculation heavy. 
        return torch.matmul(torch.matmul(bases, torch.linalg.inv(
            torch.matmul(torch.transpose(bases, 0, 1), bases) )), torch.transpose(bases, 0, 1))

    def matr_to_origin(self, tens, indx):
        """Shifts a matrix to the origin by subtracting the column at the given index.

        Args:
          tens: A tensor representing the matrix.
          indx: The index of the column to subtract.

        Returns:
          A tensor representing the shifted matrix.
        """
        # Get the shape of the tensor.
        [dim1, dim2] = tens.shape
        # Extract the column at the given index.
        t = tens[:, indx]
        # Resize the column to match the shape of the tensor.
        t = t.resize(dim1, 1).expand(dim1, dim2)
        # Subtract the column from the tensor and return the result.
        return tens-t

    def matr_vec_to_origin(self, tens, vec):
        """Shifts a matrix to the origin by subtracting a given vector.

        Args:
          tens: A tensor representing the matrix.
          vec: A tensor representing the vector to subtract.

        Returns:
          A tensor representing the shifted matrix.
        """
        # Get the shape of the tensor.
        [dim1, dim2] = tens.shape
        # Resize the vector to match the shape of the tensor.
        vec = vec.resize(dim1, 1).expand(dim1, dim2)
        # Subtract the vector from the tensor and return the result.
        return tens-vec

    def matr_zero_mean(self, M):
        """Shifts a matrix to have zero mean by subtracting the mean of each row.

        Args:
          M: A tensor representing the matrix.

        Returns:
          A tensor representing the shifted matrix.
        """
        # Get the shape of the matrix.
        [dim1, dim2] = M.shape
        # Calculate the mean of each row.
        mn = torch.mean(M, 1)
        # Resize the mean vector to match the shape of the matrix.
        mn = mn.resize(dim1, 1).expand(dim1, dim2)
        # Subtract the mean vector from the matrix and return the result.
        return M-mn
        
    def calc_mi(self, m, ind,rdim, nnum):
        """ Calculates the mutual information (MI) between a matrix and a specific column.
        
        This method shifts the matrix to the origin by subtracting the column at the given index,
        calculates the squared distances from the origin, sorts the distances, and selects the 
        `nnum` nearest neighbors. It then calculates the MI based on the selected neighbors 
        and returns the MI, singular values, and indices of the neighbors.
        
        Args:
          m: A tensor representing the matrix.
          ind: The index of the column used for shifting and neighbor selection.
          rdim: The reduced dimension for MI calculation.
          nnum: The number of nearest neighbors to consider.
          
        Returns:
          tuple: A tuple containing the MI, singular values, and indices of the nearest neighbors.
        """
        # Shift the matrix to the origin by subtracting the column at the given index.
        mto = self.matr_to_origin(m, ind)
        # Calculate the squared distances from the origin for each column.
        dig = torch.diag(torch.matmul(torch.transpose(mto, 0, 1), mto))
        # Sort the distances and get the indices of the sorted distances.
        salad, ind = torch.sort(dig)
        # Select the indices of the `nnum` nearest neighbors (excluding the origin itself).
        ind_e = ind[1:nnum+1]
        # Extract the columns corresponding to the nearest neighbors and shift them to have zero mean.
        mi= self.matr_zero_mean(m[:, ind_e])
        # Perform singular value decomposition (SVD) on the zero-mean matrix.
        u, s, vh = torch.linalg.svd(mi, full_matrices=True)
        # Calculate the MI based on the SVD results and return it along with singular values and neighbor indices.
        return( torch.matmul(u[:,:rdim], torch.diag(s[:rdim])), s[:rdim], ind_e)
    
    def curv_calc(self, m, vct, rdim, nnum):
        """Calculates the curvature between a matrix and a vector.

        This method calculates the curvature by finding the angle between the subspaces spanned by
        the nearest neighbors of the vector in the matrix and the subspace spanned by the vector
        and its nearest neighbors.

        Args:
            m: The matrix of data points.
            vct: The vector to calculate the curvature from.
            rdim: The reduced dimension for subspace calculation.
            nnum: The number of nearest neighbors to consider.

        Returns:
            The angle between the subspaces, representing the curvature.
        """
        # Shift the matrix to the origin by subtracting the vector.
        mtp = self.matr_vec_to_origin(m, vct)
        # Calculate the squared distances from the origin for each column.
        dig = torch.diag(torch.matmul(torch.transpose(mtp, 0, 1), mtp))
        # Sort the distances and get the indices of the sorted distances.
        salad, ind = torch.sort(dig)
        # Select the indices of the `nnum` nearest neighbors.
        ind_e = ind[0:nnum]
        # Extract the columns corresponding to the nearest neighbors.
        mn = m[:, ind_e]
        # Shift the nearest neighbors to have zero mean.
        mni = self.matr_zero_mean(mn)
        # Concatenate the nearest neighbors and the vector.
        mtot = torch.concat((mn, vct), 1)
        # Shift the concatenated matrix to have zero mean.
        mtoti = self.matr_zero_mean(mtot)
        # Perform singular value decomposition (SVD) on the zero-mean matrices.
        un, sn, vn = torch.linalg.svd(mni, full_matrices = True)
        utot, stot, vtot = torch.linalg.svd(mtoti, full_matrices = True)
        # Calculate the reduced dimension subspaces using SVD results.
        usn = torch.matmul(un[:,:rdim], torch.diag(sn[:rdim]))
        ustot = torch.matmul(utot[:,:rdim], torch.diag(stot[:rdim]))
        # Calculate the transformation matrix between the subspaces.
        Q = torch.matmul(torch.transpose(usn, 0, 1), ustot)
        # Perform SVD on the transformation matrix.
        uq, sq, vq = torch.linalg.svd(Q)
        # Calculate the angle between the subspaces using the singular values.
        theta = torch.acos(torch.abs(torch.clamp(torch.sum(sq)/torch.sum(sn[:rdim]*stot[:rdim]), min=-1.0, max=1.0)))
        # Return the angle as the curvature.
        return theta

    def curv_calc_auto(self,m, vct):
        """Calculates the curvature between a matrix and a vector with automatic dimension selection.

        This method calculates the curvature similar to `curv_calc`, but automatically selects
        the reduced dimension based on the energy threshold `self.auto_threshold`. It iterates
        through different numbers of nearest neighbors and dimensions, stopping when the dimension
        stops increasing.

        Args:
            m: The matrix of data points.
            vct: The vector to calculate the curvature from.

        Returns:
            The angle between the subspaces, representing the curvature.
        """
        # Shift the matrix to the origin by subtracting the vector.
        mtp = self.matr_vec_to_origin(m, vct)
        # Calculate the squared distances from the origin for each column.
        dig = torch.diag(torch.matmul(torch.transpose(mtp, 0, 1), mtp))
        # Sort the distances and get the indices of the sorted distances.
        salad, ind = torch.sort(dig)

        # Initialize lists to store the number of neighbors and corresponding dimensions.
        neighbour_number_list = list()
        neighbour_dim_list = list()

        # Iterate through different numbers of nearest neighbors.
        for indx_num in range(2,100):
            # Select the indices of the current number of nearest neighbors.
            ind_e = ind[0:indx_num]
            # Extract the columns corresponding to the nearest neighbors.
            mn = m[:, ind_e]
            # Shift the nearest neighbors to have zero mean.
            mni = self.matr_zero_mean(mn)
            # Concatenate the nearest neighbors and the vector.
            mtot = torch.concat((mn, vct), 1)
            # Shift the concatenated matrix to have zero mean.
            mtoti = self.matr_zero_mean(mtot)

            # Perform singular value decomposition (SVD) on the zero-mean matrices.
            un, sn, vn = torch.linalg.svd(mni, full_matrices = True)
            utot, stot, vtot = torch.linalg.svd(mtoti, full_matrices = True)

            # Calculate the cumulative energy of singular values.
            energy_tensor = torch.cumsum(sn, dim = 0)/torch.sum(sn, dim = 0)
            # Determine the reduced dimension based on the energy threshold.
            try:
                rdim = torch.min((energy_tensor > self.auto_threshold).nonzero().squeeze()).item()
            except:
                rdim = 0
            # Store the number of neighbors and the corresponding dimension.
            neighbour_number_list.append(indx_num)
            neighbour_dim_list.append(rdim)
            # Stop iterating if the dimension stops increasing.
            if 3<=indx_num:
                if rdim<=neighbour_dim_list[-2]:
                    break
        # Calculate the reduced dimension subspaces using SVD results.
        usn = torch.matmul(un[:,:rdim], torch.diag(sn[:rdim]))
        ustot = torch.matmul(utot[:,:rdim], torch.diag(stot[:rdim]))
        # Calculate the transformation matrix between the subspaces.
        Q = torch.matmul(torch.transpose(usn, 0, 1), ustot)
        # Perform SVD on the transformation matrix.
        uq, sq, vq = torch.linalg.svd(Q)
        # Calculate the angle between the subspaces using the singular values.
        theta = torch.acos(torch.abs(torch.clamp(torch.sum(sq)/torch.sum(sn[:rdim]*stot[:rdim]), min=-1.0, max=1.0)))
        # Return the angle as the curvature.
        return theta

    def neighbour_projection(self, m, vct):
        mtp = self.matr_vec_to_origin(m, vct)

    def optimal_thresh(self, prjct_matr, origin_zero, non_origin_zero):
        prjcts = torch.matmul(prjct_matr, origin_zero)
        non_prjcts = torch.matmul(prjct_matr, non_origin_zero)
        ratio_vls = torch.div(torch.norm(prjcts, dim = 0), torch.norm(origin_zero, dim = 0))
        non_ratio_vls = torch.div(torch.norm(non_prjcts, dim = 0), torch.norm(non_origin_zero, dim = 0))

        ref_range = 100
        true_pos_rat = []
        false_pos_rat = []
        ref_list = []
        opt_list = []

        for k in range(ref_range):
            ref_val = k/ref_range
            ref_list.append(ref_val)
            true_pos = torch.sum(ratio_vls>ref_val).item()
            false_neg = torch.sum(ratio_vls<=ref_val).item()

            true_neg = torch.sum(non_ratio_vls<=ref_val).item()
            false_pos = torch.sum(non_ratio_vls>ref_val).item()

            tpr = true_pos/(true_pos+false_neg)
            true_pos_rat.append(tpr)
            fpr = false_pos/(false_pos+true_neg)
            false_pos_rat.append(fpr)
            opt_list.append((fpr**2)+(1-tpr)**2)
            #print('tp, fn, tn, fp:',true_pos, false_neg, true_neg, false_pos)
        save_thresh = ref_list[opt_list.index(min(opt_list))]
        return(ratio_vls, non_ratio_vls, ref_list, opt_list, save_thresh, true_pos_rat, false_pos_rat)
    
    def get_subspace_bases(self):
        stat_tens, re_tens, nam_list = self.get_tens()
        state_tens = stat_tens[:,:8000]
        res_tens = re_tens[:8000,:]
        name_list = nam_list[:8000]

        #print(res_tens.shape)
        #taking transpose of the state tensor to make a column matrix
        # all results are looked up and affordance classes are listed
        self.afford_labellist = res_tens.unique().tolist()
        # '0' is removed since all results include it
        self.afford_labellist.remove(0)
        # defining affordance label value to name dictionary

        for i in range(len(self.afford_labellist)):
            self.afford_dict[self.afford_labellist[i]] = self.affordance_names[i]
            self.afford_dict_T[self.afford_labellist[i]] = self.affordance_names_T[i]

        non_base_point_vecs = dict()
        non_state_dict = dict()
        ratio_vals = dict()
        non_ratio_vals = dict()
        nr_of_bases_dict = dict()
        subs_angls = dict()


        red_dim = 3
        num_n = 10
        angl_list_tot = []
        if self.plot_graph:
            plt.figure()
        for i in self.afford_labellist:
            indices = torch.nonzero(torch.sum( (res_tens == i).int() , axis = 1))
            non_indices = torch.nonzero(torch.sum( (res_tens == i).int() , axis = 1)==0)
            afford_states = state_tens[:, indices.squeeze()].to(self.device)
            non_afford_states = state_tens[:, non_indices.squeeze()].to(self.device)
            self.state_dict[i] = afford_states

            mean_val = torch.mean(afford_states,1)
            non_mean_val = torch.mean(non_afford_states, 1).to(self.device)
            self.base_point_vecs[i] = mean_val.unsqueeze(1).to(self.device)
            [dim1, dim2] = afford_states.shape
            base_tens = mean_val.resize(dim1, 1).expand(dim1, dim2).to(self.device)
            origin_zero_matr = (afford_states-base_tens).to(self.device)
            U, S, Vh = torch.linalg.svd(origin_zero_matr, full_matrices = True)
            ratio_tens = torch.zeros(afford_states.shape)
            non_ratio_tens = torch.zeros(non_afford_states.shape)
            all_ratio_tens = torch.zeros(torch.cat((ratio_tens, non_ratio_tens), 1).shape)

            [dim1, dim2] = non_afford_states.shape
            non_base_origin = non_mean_val.resize(dim1, 1).expand(dim1, dim2).to(self.device)
            non_origin_tozero_matr = (non_afford_states-non_base_origin).to(self.device)


            ratio_tens = torch.abs(torch.div(torch.matmul(torch.transpose(U,0,1).to(self.device), origin_zero_matr), torch.norm(origin_zero_matr, dim = 0).unsqueeze(0).expand(origin_zero_matr.shape[0],-1)))
            non_ratio_tens = torch.abs(torch.div(torch.matmul(torch.transpose(U,0,1).to(self.device), non_origin_tozero_matr), torch.norm(non_origin_tozero_matr, dim = 0).unsqueeze(0).expand(non_origin_tozero_matr.shape[0],-1)))
            state_proj_mean = torch.mean(ratio_tens, 1)
            non_state_proj_mean = torch.mean(non_ratio_tens, 1)
            indc = torch.nonzero(state_proj_mean>non_state_proj_mean).squeeze().int()
            rtls, non_ratls, ref_list, opt_list, save_thresh, true_pos_rat, false_pos_rat = self.optimal_thresh(self.project_matr( U[:,indc]).to(self.device), origin_zero_matr, non_origin_tozero_matr)
            self.threshold_dict[i] = save_thresh
            self.base_list[i] =self.project_matr( U[:, indc]).to(self.device)
            if self.plot_graph:
                plt.plot(false_pos_rat, true_pos_rat, linewidth=3.0, label = self.afford_dict_T[i])
                plt.scatter(false_pos_rat[opt_list.index(min(opt_list))], true_pos_rat[opt_list.index(min(opt_list))], color = 'r', zorder =1000, s = 60)

        if self.plot_graph:
            plt.xlabel('False Positive Ratio')
            plt.ylabel('True Positive Ratio')
            plt.grid()
            #plt.legend()
            plt.title('ROC Curves of Affordance Groups for %s'%self.model_name_T)
            plt.savefig('ROC_Curve_%s.png'%self.model_name_T.replace("/", ""))
        return 0

    def load_models(self):
        print(self.model_name)
        self.featureExtractor = FeatureExtractorNet(model_name=self.model_name)
        self.objectDetector = ObjectDetectorYOLO()
        self.segmentator = ObjectSegmentorYOLO()

    def getWebcamPhoto(self, imagename='photo.jpg'):
        self.take_photo(imagename)

    def get_opt_result(self, scores, threshold_Val):
        score_dict = dict()
        scores_sorted, scores_indices = torch.sort(scores)
        w_ordered = [self.afford_dict[self.afford_labellist[x]] for x in scores_indices.tolist()]
        out_max = torch.max(scores)
        out_min = torch.min(scores)
        output_w = (scores - out_min*torch.ones_like(scores))/(out_max-out_min)
        outp, outi = torch.sort(output_w)
        range_tens = torch.range(0, len(self.afford_labellist)-1.0)/(len(self.afford_labellist)-1.0)
        out_optim = torch.square(range_tens-torch.ones_like(range_tens)) + torch.square(outp)
        optim_sorted, optim_ind_sorted = torch.sort(out_optim)
        if self.plot_graph:
            plt.figure()
            #plt.plot(range_tens.tolist(), outp.tolist(), linewidth=3.0)
            for k in range(optim_ind_sorted.shape[0]):
                #print('at position:',range_tens[k].item(), optim_sorted[k].item(), 'affordance ', self.afford_dict[self.afford_labellist[optim_ind_sorted[k]]])
                if k ==optim_ind_sorted[0]:
                    plt.text(range_tens[k].item()+0.02, outp[k].item(), self.afford_dict[self.afford_labellist[outi[k]]], color='red', rotation = 60)
                    plt.scatter(range_tens[k].item(), outp[k].item(), color='red', zorder =1000, s = 60)
                else:
                    plt.text(range_tens[k].item()+0.02, outp[k].item(), self.afford_dict[self.afford_labellist[outi[k]]], color='blue', rotation = 60)
                    plt.scatter(range_tens[k].item(), outp[k].item(), color='blue', zorder =1000, s = 60)
            plt.show()

        w_results = torch.nonzero(output_w >= outp[optim_ind_sorted[0]]).squeeze()
        mults = (output_w >= 1.2*outp[optim_ind_sorted[0]]).squeeze()*output_w.squeeze()
        mults[mults == 0] = -10e10
        mults = self.softmax(mults)
        score_dict['mult'] = mults
        score_dict['optim_result'] = [self.afford_dict[self.afford_labellist[k]] + ' %.3f'%mults[k] for k in w_results.tolist()]
        return score_dict


    def image_estimate(self, imagename='photo.jpg'):
        img = cv2.imread( imagename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        crops = self.objectDetector.detect_objects(pil_img)
        result_seg = self.segmentator.detect_objects(imagename)
        TINT_COLOR = (0, 0, 0)  # Black
        TRANSPARENCY = .35  # Degree of transparency, 0-100%
        OPACITY = int(255 * TRANSPARENCY)

        masks_seg = result_seg.masks.data
        boxes_seg = result_seg.boxes.data
        print('boxes seg', boxes_seg)
        clss_seg = boxes_seg[:, 5]
        print('clss seg', clss_seg)
        print('Segmented results are (YOLOv8):')
        crop_list = list()
        crop_labels = list()
        crop_centers = list()
        for itr in range(len(clss_seg.tolist())):
            clss_num = clss_seg[itr]
            if self.segmentator.yolo_classes[int(clss_num.item())] == 'person':
                pass
            else:
                cls_masks = masks_seg[itr]
                cls_mask = (cls_masks).unsqueeze(0)
                cls_mask = (transforms.Resize(size = (765, 1360))(torch.cat((cls_mask, cls_mask, cls_mask),0)) >0.5).int()
                cls_mask = cls_mask.permute(1,2,0)
                orig_im = torch.tensor(result_seg.orig_img)
                orig_im = orig_im[:,:,[2,1,0]]
                zeros_im = torch.zeros_like(orig_im)
                masked_img = torch.where(cls_mask == 1, orig_im, zeros_im)
                x0, y0, x1, y1 = boxes_seg[itr, :4].int()
                width = masked_img.shape[0]
                height = masked_img.shape[1]
                x0 = int(x0*0.9)
                y0 = int(y0*0.9)
                x1 = min(int(x1*1.1), height)
                y1 = min(int(y1*1.1), width)
                cropped_m = masked_img[y0:y1, x0:x1, :]
                cropped = Image.fromarray(cropped_m.numpy())
                crop_t = self.featureExtractor.extract_features(cropped)
                v1 = crop_t.squeeze(3).squeeze(0).to(self.device)
                print((v1-self.base_point_vecs[1]).shape)

                prjctns = torch.tensor([(torch.norm(torch.matmul(self.base_list[x], v1-self.base_point_vecs[x]))/torch.norm(v1-self.base_point_vecs[x])).item()-0.85*self.threshold_dict[x] for x in self.afford_labellist])
                prjctns = torch.abs(prjctns)
                prj_dict = self.get_opt_result(prjctns, 0.5)

                ang_la1 = [self.curv_calc(self.state_dict[k], v1, 3,20).item() for k in self.afford_labellist]
                la1_dict = self.get_opt_result(1/torch.tensor(ang_la1), 1.7)

                w_weighted = torch.matmul(self.W_matr.float(), la1_dict['mult'].unsqueeze(1))+ torch.matmul(self.W_matr.float(), prj_dict['mult'].unsqueeze(1))
                #output_w = m((1/w_weighted).squeeze())
                w_weighted = (1.0/w_weighted).squeeze()
                w_dict = self.get_opt_result(w_weighted, 0.000000000000000001)
                w_tot_results = w_dict['optim_result']
                if len(w_tot_results) == 0:
                    w_tot_results = ['found none']
                box_coords = boxes_seg[itr, :4].squeeze().tolist()
            cv2_img = cv2.imread( imagename)
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            source_img = Image.fromarray(cv2_img).convert("RGBA")
            overlay = Image.new('RGBA', source_img.size, TINT_COLOR+(0,))
            draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
            draw.rectangle(((x0, y0), (x1, y1)), outline =(255, 0, 0), width = 5, fill=TINT_COLOR+(OPACITY,))
            print_string =  w_tot_results
            print_string = '\n'.join(print_string)
            draw.text((box_coords[0], box_coords[1]), 'YOLOv8 est is:' + self.segmentator.yolo_classes[int(clss_num.item())], font =ImageFont.truetype('LiberationMono-Bold.ttf',25),fill = (0, 0, 255, 255))
            draw.multiline_text((box_coords[0], box_coords[1]+20), print_string, font =ImageFont.truetype('LiberationMono-Bold.ttf',25),fill = (255, 255, 0, 255))
            source_img = Image.alpha_composite(source_img, overlay)
            source_img = source_img.convert("RGB") # Remove alpha for saving in jpg format.

            plt.imshow(cropped)
            plt.show()

            plt.imshow(source_img)
            plt.show()


        crops = self.objectDetector.detect_objects(pil_img)

        print('Non-Segmented Results are(YOLOv5):')

        for crop in crops:
            if 'person' in crop['label']:
                pass
            else:
                #since yolo sometimes crops the corners of the objects, lets make crop region 20% bigger
                x0, y0, x1, y1 = crop['box']
                width, height = pil_img.size
                x0 = int(x0*0.9)
                y0 = int(y0*0.9)
                x1 = min(int(x1*1.1), width)
                y1 = min(int(y1*1.1), height)
                cropped = pil_img.crop((x0, y0, x1, y1))
                #display(Image(crop))
                #crop_im = Image.fromarray(crop['im'][:,:,::-1])
                crop_t = self.featureExtractor.extract_features(cropped)
                v1 = crop_t.squeeze(3).squeeze(0).to(self.device)

                prjctns = torch.tensor([(torch.norm(torch.matmul(self.base_list[x], v1-self.base_point_vecs[x]))/torch.norm(v1-self.base_point_vecs[x])).item()-0.85*self.threshold_dict[x] for x in self.afford_labellist])
                prjctns = torch.abs(prjctns)
                prj_dict = self.get_opt_result(prjctns, 0.5)


                ang_la1 = [self.curv_calc(self.state_dict[k], v1, 3,20).item() for k in self.afford_labellist]
                la1_dict = self.get_opt_result(1/torch.tensor(ang_la1), 1.7)


                w_weighted = torch.matmul(self.W_matr.float(), la1_dict['mult'].unsqueeze(1))+ torch.matmul(self.W_matr.float(), prj_dict['mult'].unsqueeze(1))
                #output_w = m((1/w_weighted).squeeze())
                w_weighted = (1.0/w_weighted).squeeze()
                w_dict = self.get_opt_result(w_weighted, 0.000000000000000001)
                w_tot_results = w_dict['optim_result']
                if len(w_tot_results) == 0:
                    w_tot_results = ['found none']
                box_coords = [x.item() for x in crop['box']]
                cv2_img = cv2.imread( imagename)
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                source_img = Image.fromarray(cv2_img).convert("RGBA")
                overlay = Image.new('RGBA', source_img.size, TINT_COLOR+(0,))
                draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
                draw.rectangle(((x0, y0), (x1, y1)), outline =(255, 0, 0), width = 5, fill=TINT_COLOR+(OPACITY,))
                print_string =  w_tot_results
                print_string = '\n'.join(print_string)
                draw.text((box_coords[0], box_coords[1]), 'YOLOv5 est is:'+ crop['label'], font =ImageFont.truetype('LiberationMono-Bold.ttf',25),fill = (0, 0, 255, 255))
                draw.multiline_text((box_coords[0], box_coords[1]+20), print_string, font =ImageFont.truetype('LiberationMono-Bold.ttf',25),fill = (255, 255, 0, 255))
                source_img = Image.alpha_composite(source_img, overlay)
                source_img = source_img.convert("RGB") # Remove alpha for saving in jpg format.

                plt.imshow(source_img)
                plt.show()
