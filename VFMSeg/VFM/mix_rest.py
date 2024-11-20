import torch
import numpy as np
import time






# Val SAM
    # # Corresponding Point Indices within Image Pixel (Num, True_or_False)
    # sampled_2d_to_3d_indices = np.zeros(sampled_2d_masks.shape,dtype=np.bool8)
    # sampled_2d_to_3d_indices[indices[:,0],indices[:,1]] = True
    
    # unique_indices, counts = np.unique(indices,axis=0,return_counts=True)
    # # n1 = len(unique_indices[counts>1])   # 35
    # # n0 = len(unique_indices[counts==1])  # 2847
    # # # 2917 - 2882 = 35    2937-2903 = 34
    # # # 2847 + 35 = 2882    2869 + 34 =2903
    # seq = 0
    # t = 0
    # war = 0
    # for hw in unique_indices[counts>1]:
    #     # print('/n Seq: ',seq)
    #     tmp = []
    #     for idx in indices:
    #         if hw[0] == idx[0] and hw[1] == idx[1]:
    #             print(seg_label[seq])
    #             tmp.append(seg_label[seq])
    #             print(coords[seq])
    #     # print('Repeat cnt:', len(tmp))
    #     if len(tmp) > 2:
    #         # print('More than two recurrence.')
    #         t = 1

    #     else:
    #         if tmp[0] == tmp[1]:
    #             # print('Consistent Label.')
    #             t = 1
    #         else:
    #             war += 1
    #             # print('Warning Ambiguous Label!')
    #     seq +=1

    # if war > 0:
    #     print('Ambiguous Label Found!')

    # # Validate Indices
    # num = np.count_nonzero(sampled_2d_to_3d_indices == np.bool8(True))
    # idx = 0
    # Count_Indices_True = 0
    # for corrd in indices:
    #     if sampled_2d_to_3d_indices[corrd[0],corrd[1]] is np.bool8(True):
    #         Count_Indices_True += 1
    #     else:
    #         print('Missing Point Index: ',idx)
    #     idx +=1

    # Count_Indices_True_1 = 0
    # for h in range(sampled_2d_masks.shape[0]):
    #     for w in range(sampled_2d_masks.shape[1]):
    #         if sampled_2d_to_3d_indices[h,w] is np.bool8(True):
    #             Count_Indices_True_1 += 1


    # Count_Indices_Sample_True_1 = 0
    # for corrd in indices:
    #     if sampled_2d_to_3d_indices[corrd[0],corrd[1]] is np.bool8(True) and sampled_2d_masks[corrd[0],corrd[1]] is np.bool8(True):
    #         Count_Indices_Sample_True_1 += 1
    
    # Count_Indices_Sample_True_2 = 0
    # for corrd in indices:
    #     if sampled_2d_masks[corrd[0],corrd[1]] is np.bool8(True):
    #         Count_Indices_Sample_True_2 += 1

    # sampled_2d_to_3d_indices = sampled_2d_to_3d_indices & sampled_2d_masks

    # # Validate Indices
    # Count_Sample_True = 0
    # Count_Mask_True = 0
    # for h in range(sampled_2d_masks.shape[0]):
    #     for w in range(sampled_2d_masks.shape[1]):
    #         if sampled_2d_masks[h,w] == np.bool8(True):
    #             Count_Mask_True += 1
            
    #         if sampled_2d_to_3d_indices[h,w] == np.bool8(True):
    #             Count_Sample_True += 1
    

    # sampled_indice_list = sampled_2d_to_3d_indices[indices[:,0],indices[:,1]] == True

    # # Validate
    # Count_Sample_Indice_True = 0
    # for e in sampled_indice_list:
    #     if e == True:
    #         Count_Sample_Indice_True += 1


    # sampled_indice_list = sampled_2d_masks[indices[:,0],indices[:,1]] == True
    # # Validate
    # Count_Sample_Indice_True_2 = 0
    # for e in sampled_indice_list:
    #     if e ==True:
    #         Count_Sample_Indice_True_2 += 1



# val Cut
    # # Corresponding Point Indices within Image Pixel
    # cut_2d_to_3d_indices = np.zeros((h, w),dtype=np.bool8)
    # cut_2d_to_3d_indices[indices[:,0],indices[:,1]] = True
    
    # # Validate
    # # num = np.count_nonzero(cut_2d_to_3d_indices == np.bool8(True))
    # # idx = 0
    # # Count_True = 0
    # # for corrd in indices:
    # #     if cut_2d_to_3d_indices[corrd[0],corrd[1]] is np.bool8(True):
    # #         Count_True += 1
    # #     else:
    # #         print('Missing Point Index: ',idx)
    # #     idx +=1

    # cut_2d_to_3d_indices = cut_2d_to_3d_indices & mask

    # # Validate Indices
    # # Count_Sample_True = 0
    # # Count_Mask_True = 0
    # # for h in range(mask.shape[0]):
    # #     for w in range(mask.shape[0]):
    # #         if mask[h,w] is np.bool8(True):
    # #             Count_Mask_True += 1
            
    # #         if cut_2d_to_3d_indices[h,w] is np.bool8(True):
    # #             Count_Sample_True += 1