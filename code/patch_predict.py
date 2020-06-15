import numpy as np
from sklearn.linear_model import Lasso
from img_tools import patch_to_vector,vector_to_patch
from sklearn.metrics import mean_squared_error as MSE
from img_tools import channel_recover 


'''Establish the lasso model corresponding to the three RGB channels'''
def get_lasso_models(dictionnaire, patch_noise, alpha=0.01, max_iter=1000):
    '''
        (np.array)*(np.array)*(float)*(int) => (np.array)
        dictionnaire: All patches without noise (including x_train and x_test): take x_train here
        patch_noise: Patch with noise (including y_train)
        alpha: Regular parameters needed to build the lasso model (appropriate reduction will help alleviate underfitting, too small will cause overfitting): the default is 0.01
        max_iter: The maximum number of iterations specified when building the lasso model: the default is 1000
    '''
    models = []
    dic = []
    for patch in dictionnaire:
        dic.append(patch_to_vector(patch))
    dic = np.array(dic)
   
    p_n = patch_to_vector(patch_noise)
    x_train_r,x_train_g,x_train_b,y_train_r,y_train_g,y_train_b = [],[],[],[],[],[]
    
    noise = [-100.,-100.,-100.]
    for n in range(p_n.shape[0]):
        if (p_n[n]-noise).any() == True:
            x_train_r.append(dic[:,n,0])
            x_train_g.append(dic[:,n,1])
            x_train_b.append(dic[:,n,2])
            y_train_r.append(p_n[n,0])
            y_train_g.append(p_n[n,1])
            y_train_b.append(p_n[n,2])
            
    model_r = Lasso(alpha=alpha,max_iter=max_iter).fit(x_train_r, y_train_r)
    model_g = Lasso(alpha=alpha,max_iter=max_iter).fit(x_train_g, y_train_g)
    model_b = Lasso(alpha=alpha,max_iter=max_iter).fit(x_train_b, y_train_b)
    
    models.append(model_r)
    models.append(model_g)
    models.append(model_b)
    
    return np.array(models)


'''Use the lasso model to make predictions'''
def repair_predict(models, dictionnaire, patch_noise):
    '''
        (np.array)*(np.array)*(np.array) => (np.array)
        models: Contains three lasso models corresponding to RGB
        dictionnaire: All patches without noise (including x_train and x_test): take x_test here
        patch_noise: Patch with noise (including y_train)
    '''
    dic = []
    for patch in dictionnaire:
        dic.append(patch_to_vector(patch))
    dic = np.array(dic)
        
    p_n = patch_to_vector(patch_noise)
    x_test_r,x_test_g,x_test_b = [],[],[]
    
    noise = [-100.,-100.,-100.]
    for n in range(p_n.shape[0]):
        if (p_n[n]-noise).any() == False:
            x_test_r.append(dic[:,n,0])
            x_test_g.append(dic[:,n,1])
            x_test_b.append(dic[:,n,2])
            
    y_predict_r = models[0].predict(x_test_r)
    y_predict_g = models[1].predict(x_test_g)
    y_predict_b = models[2].predict(x_test_b)
    
    y_predict_rgb = list(zip(y_predict_r,y_predict_g,y_predict_b))
    y_predict_rgb = np.array([list(rgb) for rgb in y_predict_rgb])
    
    return y_predict_rgb



'''Patch patches containing noise'''
def repatching(y_predict_rgb, patch_noise,h):
    '''
        (np.array)*(np.array)*(int) => (np.array)
        y_predict_rgb: Predicted value of RGB three channels at noise
        patch_noise: Patch with noise (including y_train)
        h: patch width and height
    '''
    noise = [-100.,-100.,-100.]
    p_n = patch_to_vector(patch_noise)
    i=0
    for n in range(p_n.shape[0]):
        if (p_n[n]-noise).any() == False:
            p_n[n] = y_predict_rgb[i]
            i=i+1
    p_n = vector_to_patch(p_n, h)
    return channel_recover(p_n)
        
        
'''Evaluate the prediction results'''
def repair_score(y_test_rgb, y_predict_rgb):
    '''
        (np.array)*(np.array) => (float)
        y_predict_rgb: Predicted value of RGB three channels at noise
        y_test_rgb: The true value of RGB three channels at noise
    '''
    mse_r = MSE(y_predict_rgb[:,0], y_test_rgb[:,0])
    mse_g = MSE(y_predict_rgb[:,1], y_test_rgb[:,1])
    mse_b = MSE(y_predict_rgb[:,2], y_test_rgb[:,2])
    return (mse_r+mse_g+mse_b)/3


'''Calculation error rate'''
def error_rate(score, mean):
    '''
        (float)*(float) => (float)
        score: score
        mean: Mean of true value
    '''
    return '%.2f%%' % ((score/mean)*100)


'''Get the true and predicted values on the full picture'''
def get_y_test_and_predict(img_array_noise,img_array,img_predict):
    '''
        (np.array)*(np.array)*(np.array) => ((np.array)*(np.array))
        img_array_noise: Noise image
        img_array: Original image
        img_predict: Repair complete image
    '''
    y_test = []
    y_predict = []
    noise = [-100,-100,-100]
    for i in range(img_array_noise.shape[0]):
        for j in range(img_array_noise.shape[1]):
            if (img_array_noise[i,j]-noise).any() == False:
                y_test.append(img_array[i,j])
                y_predict.append(img_predict[i,j])
    return np.array(y_test), np.array(y_predict)