import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


'''read image'''
def read_img (file_img):
    '''
        (string) => (np.array)
        file_img: Image path
    '''
    return plt.imread(file_img)


'''Convert RGB image to HSV image'''
def rgb_to_hsv(img_array_rgb):
    '''
        (np.array) => (np.array)
        img_array_rgb: Each element is a two-dimensional image matrix (np.int) of an array of size 3 (stores the values of three RGB channels)
    '''
    return matplotlib.colors.rgb_to_hsv(img_array_rgb.astype(np.int32))


'''Convert HSV image to RGB image'''
def hsv_to_rgb(img_array_hsv):
    '''
        (np.array) => (np.array)
        img_array_rgb: Each element is a two-dimensional image matrix (np.float) of an array of size 3 (stores the values of the three channels of HSV)
    '''
    return matplotlib.colors.hsv_to_rgb(img_array_hsv).astype(np.int32)


'''display an image'''
def display_img(img_array,title=None):
    '''
        (np.array) => (None)
        img_array: Image two-dimensional matrix  
    '''
    img = img_array.copy()
    img = np.where(img==[-100,-100,-100],[0,0,0],img)
    if title != None:
        plt.title(title)
    plt.imshow(img)


'''display some images once'''
def display_imgs(img_arrays,titles,h,w):
    '''
        (np.array)*(string array)*(int)*(int) => (None)
        img_array: Image two-dimensional matrix  
        titles: Image titles
        h: Number in vertical direction
        w: Number in the horizontal direction
    '''
    for i in range(len(img_arrays)):
        plt.subplot(h,w,i+1)
        display_img(img_arrays[i],titles[i])
    plt.show()

        
        
'''Standardize the image channel to improve the accuracy of the built model'''
def channel_normalize(img_array):
    '''
        (np.array) => (np.array)
        img_array: Image two-dimensional matrix
    '''
    img_array_rgb = img_array
    if(img_array.dtype == 'float32' or img_array.dtype == 'float64'):
        img_array_rgb = hsv_to_rgb(img_array)
    return np.where(img_array_rgb==[-100,-100,-100],img_array_rgb,img_array_rgb/256)


'''Channel restoration of the standardized image for easy display'''
def channel_recover(img_array):
    '''
        (np.array) => (np.array)
        img_array: Image two-dimensional matrix
    '''
    return np.where(img_array==[-100,-100,-100],img_array,img_array*256).astype(np.int32)


'''Get a patch with i, j as the center and width and height as h'''
def get_patch(i,j,h,img_array):
    '''
        (int)*(int)*(int)*(np.array) => (np.array)
        i,j: patch center coordinates
        h: patch width and height
        img_array: Image two-dimensional matrix
    '''
    height = img_array.shape[0]
    width = img_array.shape[1]
    if(i<h/2-1 or j<h/2-1 or i>height-h/2 or j>width-h/2):
        print("i or j out of range !!!")
    else:
        row_start = int(i-np.floor(h/2))
        row_end = int(i+np.ceil(h/2))
        col_start = int(j-np.floor(h/2))
        col_end = int(j+np.ceil(h/2))
        return img_array[row_start:row_end,col_start:col_end]
    

'''Get all the patches on the image, and the center coordinates corresponding to each patch'''
def get_all_patch(h,img_array,d_h=1,d_w=1):
    '''
        (int)*(np.array)*(int)*(int) => ((np.array)*(np.array))
        h: patch width and height
        img_array: Image two-dimensional matrix
        d_h: The distance between different patches in the height direction (default is 1)
        d_w: Spacing in the width direction between different patches (default is 1)
    '''
    height = img_array.shape[0]
    width = img_array.shape[1]
    if (height-h)%d_h != 0 or (width-h)%d_w != 0:
        print("distance between patchs not match !!!")
    else:
        start_i = int(np.ceil(h/2-1))
        end_i = int(np.ceil(height-h/2))
        start_j = int(np.ceil(h/2-1))
        end_j = int(np.ceil(width-h/2))
        all_patch = []
        all_patch_coord = []
        for i in range(start_i, end_i, d_h):
            for j in range(start_j, end_j, d_w):
                all_patch.append(get_patch(i,j,h,img_array))
                all_patch_coord.append([i,j])
        return np.array(all_patch),np.array(all_patch_coord)


'''Get all the noisy patches on the image, and the center coordinates where it is, that is, the Y training set, including y_train'''
def get_patch_bruite(all_patchs):
    '''
        ((np.array)*(np.array)) => ((np.array)*(np.array))
        all_patchs: All patches of the image and its coord
    '''
    all_patch,all_patch_coord = all_patchs
    all_patch_bruite = []
    patch_bruite_coord = []
    for i in range(all_patch.shape[0]):
        if([-100,-100,-100] in all_patch[i]):
            all_patch_bruite.append(all_patch[i])
            patch_bruite_coord.append(all_patch_coord[i])
    return np.array(all_patch_bruite),np.array(patch_bruite_coord)

    
'''Get all patches without noise on the image, namely the training set and test set of X, including x_train and x_test'''
def get_dictionnaire(all_patchs):
    '''
        ((np.array)*(np.array)) => (np.array)
        all_patchs: All patches of the image and its coord
    '''
    all_patch = all_patchs[0]
    dictionnaire = []
    for i in range(all_patch.shape[0]):
        if([-100,-100,-100] not in all_patch[i]):
            dictionnaire.append(all_patch[i])
    return np.array(dictionnaire)
            

'''Convert the patch into a one-dimensional vector for easy calculation'''
def patch_to_vector(patch_array):
    '''
        (np.array) => (np.array)
        patch_array: hxh two-dimensional matrix storing patch
    '''
    h = patch_array.shape[0]
    return patch_array.reshape(h*h,3)


'''Convert the vector back to patch to facilitate completion of the image after the calculation'''
def vector_to_patch(vector, h):
    '''
        (np.array)*(int) => (np.array)
        vertor: One-dimensional array of storing patch
        h: patch width and height
    '''
    if(vector.shape[0] != h*h):
        print("This vector has wrong size, it can not be transformed to patch !!!")
    else:
        return vector.reshape(h,h,3)
    
    
'''Add random noise to the image'''
def make_noise(img_array, prc):
    '''
        (np.array)*(float) => (np.array)
        img_array: Image two-dimensional matrix
        prc: Noise percentage
    '''
    img_array = np.require(img_array, dtype='f4', requirements=['O', 'W']).astype(np.int32)
    nb_row = img_array.shape[0]
    nb_col = img_array.shape[1]
    nb_noise = int(nb_row*nb_col*prc)
    row_index = np.arange(img_array.shape[0])
    col_index = np.arange(img_array.shape[1])
    coord = np.transpose([np.tile(row_index, nb_col), np.repeat(col_index, nb_row)])
    coord_index = np.random.choice(coord.shape[0],nb_noise,replace=False)
    coord_noise = coord[coord_index]
    coord_noise = np.hsplit(coord_noise,2)
    img_array[coord_noise[0],coord_noise[1]] = [-100,-100,-100]
    return img_array


'''Delete the square with the center i, j height h and width w on the image (simulating the missing part of the image)'''
def delete_rect(img_array,i,j,h,w):
    '''
        (np.array)*(int)*(int)*(int)*(int) => (np.array)
        img_array: Image two-dimensional matrix
        i,j: The center coordinates of the deleted square
        h: The height of the deleted square
        w: The width of the deleted square
    '''
    img_array = np.require(img_array, dtype='f4', requirements=['O', 'W']).astype(np.int32)
    
    height = img_array.shape[0]
    width = img_array.shape[1]
    
    if(i<h/2-1 or j<h/2-1 or i>width-h/2 or j>height-h/2):
        print("out of range, delete impossible !!!")
    else:
        row_start = int(i-np.floor(h/2))
        row_end = int(i+np.ceil(h/2))
        col_start = int(j-np.floor(w/2))
        col_end = int(j+np.ceil(w/2))
        
        row_index = np.arange(row_start, row_end)
        col_index = np.arange(col_start, col_end)
        nb_row = row_index.shape[0]
        nb_col = col_index.shape[0]
        
        coord = np.transpose([np.tile(row_index, nb_col), np.repeat(col_index, nb_row)])
        coord_delete = np.hsplit(coord,2)
        
        img_array[coord_delete[0],coord_delete[1]] = [-100,-100,-100]
        return img_array

    
'''Get the coordinates of noise in a patch (for result evaluation)'''
def get_noise_pixels_coords(img_array):
    '''
        (np.array) => (np.array)
        img_array: Image two-dimensional matrix
    '''
    coord_h,coord_w = [],[]
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (img_array[i,j]-[-100,-100,-100]).any()==False:
                coord_h.append(i)
                coord_w.append(j)
    return (np.array(coord_h),np.array(coord_w))
    
    
'''Get the true pixel value at the noise, that is, the test set of Y, y_test'''
def get_true_values(img_array_noise, img_array):
    '''
        (np.array)*(np.array) => (np.array)
        img_array_noise: Two-dimensional image matrix with noise
        img_array: Two-dimensional image matrix without noise
    '''
    coord = get_noise_pixels_coords(img_array_noise)
    return img_array[coord]


'''Get the pixel coordinates of the missing part of the image'''
def get_hole_pixels_coords(img_array):
    '''
        (np.array) => (np.array)
        img_array: Two-dimensional image matrix without noise
    '''
    coords = []
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (img_array[i,j]-[-100,-100,-100]).any() == False:
                coords.append([i,j])
    coords = np.array(coords)
    coords_i = list(set(coords[:,0]))
    coords_j = list(set(coords[:,1]))
    return coords.reshape(len(coords_i),len(coords_j),2)