import numpy as np
from patch_predict import get_lasso_models, repair_predict, repatching
from img_tools import read_img,display_img,channel_normalize,channel_recover,get_patch,get_all_patch
from img_tools import get_patch_bruite,get_dictionnaire,patch_to_vector,vector_to_patch,make_noise
from img_tools import delete_rect,get_noise_pixels_coords,get_true_values,get_hole_pixels_coords
import time
from random import sample


class Mpart_repair:
    def __init__(self,img_array_hole,alpha=0.01,max_iter=1000):
        self.img_array_hole = img_array_hole
        self.alpha = alpha
        self.max_iter = max_iter
        self.img_hole_normalized = []
        self.dictionnaire = []
        self.coords_directs = []
        self.start_coords = []
        self.patchs = []
        self.img_array_repaired = []


    def get_dict(self,h,d_h,d_w):
        self.img_hole_normalized = channel_normalize(self.img_array_hole)
        patchs = get_all_patch(h,self.img_hole_normalized,d_w,d_h)
        self.dictionnaire = get_dictionnaire(patchs)
    

    def spiral_order(self,matrix,step):
        m = matrix.copy()
        h = matrix.shape[0]
        w = matrix.shape[1]
        if h%step!=0 or w%step!=0:
            print("step not match !!!")
        else:
            i,j,di,dj = 0,0,0,step
            direct = 0 # 0: left_up
                    # 1: right_up
                    # 2: right_down
                    # 3: left_down
            for _ in range(int(h*w/step**2)):
                self.coords_directs.append(np.append(m[i,j],direct))
                m[i,j] = [-1,-1] # -Infinite processing of traversed elements (marked as traversed)
                next_step = m[(i+di)%h,(j+dj)%w]
                if (next_step-[-1,-1]).any() == False:
                    di, dj = dj, -di #Turn to the right
                    direct = (direct+1)%4 #Update direction of patch
                i += di
                j += dj
            self.coords_directs = np.array(self.coords_directs)
            #return np.array(self.coords_directs)
        
        
    def get_start_coords(self,step):
        self.start_coords = self.coords_directs.copy()
        offest = step-1
        for coord_direct in self.start_coords:
            if coord_direct[2] == 0:
                coord_direct[0] += offest
                coord_direct[1] += offest
            elif coord_direct[2] == 1:
                coord_direct[0] += offest
            elif coord_direct[2] == 3:
                coord_direct[1] += offest
        #return self.start_coords


    def get_each_patch(self,start_coord,img_array,h):
        patch = []
        if start_coord[2] == 0:
            patch = img_array[start_coord[0]-h+1:start_coord[0]+1,start_coord[1]-h+1:start_coord[1]+1]
        elif start_coord[2] == 1:
            patch = img_array[start_coord[0]-h+1:start_coord[0]+1,start_coord[1]:start_coord[1]+h]
        elif start_coord[2] == 2:
            patch = img_array[start_coord[0]:start_coord[0]+h,start_coord[1]:start_coord[1]+h]
        elif start_coord[2] == 3:
            patch = img_array[start_coord[0]:start_coord[0]+h,start_coord[1]-h+1:start_coord[1]+1]
        return np.array(patch)


    def repairing(self,h,step,train_max):
        self.img_array_repaired = np.array(self.img_hole_normalized.copy())
        dict_slice = self.dictionnaire
        start = time.time()
        for i in range(self.coords_directs.shape[0]):
            if (train_max!=None):
                dict_slice = np.array(sample(list(self.dictionnaire),train_max))
            patch = self.get_each_patch(self.start_coords[i],self.img_array_repaired,h)
            models = get_lasso_models(dict_slice,patch,self.alpha,self.max_iter)
            y_predict_rgb = repair_predict(models, dict_slice, patch).reshape(step,step,3)
            x = self.coords_directs[i][0]
            y = self.coords_directs[i][1]
            self.img_array_repaired[x:x+step,y:y+step] = y_predict_rgb
        end = time.time()
        times = end-start
        print("%.2f %s" % (times,"s"))
        self.img_array_repaired = channel_recover(self.img_array_repaired)
        #return channel_recover(self.img_array_repaired)
            
        
    def repair(self,step,h,d_h,d_w,train_max=None):
        hole_pixels_coords = get_hole_pixels_coords(self.img_array_hole) #Get the coordinates of the missing part
        self.spiral_order(hole_pixels_coords,step) #Perform a clockwise spiral traversal of the missing part, and obtain the coordinates of the upper right pixel of each small matrix with step as the length, and mark the direction of the corresponding patch for each coordinate:
                                                            # 0: left_up
                                                            # 1: right_up
                                                            # 2: right_down
                                                            # 3: left_down
        self.get_start_coords(step)  #Get the starting coordinate of the corresponding patch according to the direction of the coordinate mark
        self.get_dict(h,d_h,d_w)
        self.repairing(h,step,train_max) 
            #Start to repair the step*step part of the missing part in a clockwise spiral sequence, and obtain the repaired image
        #return img_array_repaired #返回修补完毕的图像（已恢复RGB三通道）