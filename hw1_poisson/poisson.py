import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import scipy.io
import cv2
import sys
import os

def extract_inner_boundary(mask):

    ## This function is used to get the mask inner_boundary from the original binary mask image using erosion
    mask = np.array(mask, dtype=np.uint8)
    inner_boundary = mask ^ cv2.erode(mask, np.ones((3, 3), np.uint8))

    return inner_boundary


def importing_laplace(source, target, omega, inner_boundary, has_neighbor):

  u = np.zeros((3,omega.shape[0]))
  for k in range(3):
    for index in range(omega.shape[0]):
      ## current location
      i, j = omega[index]
      label = has_neighbor[index]
      source_k = source[:,:,k]
      target_k = target[:,:,k]

      # laplace for source image
      source_grad_value = (label[0]==True)*(source_k[i,j]-source_k[i,j+1])+(label[1]==True)*(source_k[i,j]-source_k[i,j-1])\
          +(label[2]==True)*(source_k[i,j]-source_k[i+1,j])+(label[3]==True)*(source_k[i,j]-source_k[i-1,j])

      ## consider the dirichlet boundary condition
      if (inner_boundary[i][j])==1:
        dirichlet_condition_value = (label[0]==False)*target_k[i,j+1]+(label[1]==False)*target_k[i,j-1]\
          +(label[2]==False)*target_k[i+1,j]+(label[3]==False)*target_k[i-1,j]
      else:
        dirichlet_condition_value = 0

      u[k,index] = source_grad_value + dirichlet_condition_value

  return u


def mixing_laplace(source, target, omega, inner_boundary, has_neighbor):
  u = np.zeros((3,omega.shape[0]))
  for k in range(3):
    for index in range(omega.shape[0]):
      ## current location
      i, j = omega[index]
      label = has_neighbor[index]
      source_k = source[:,:,k]
      target_k = target[:,:,k]
      ## laplace for source image
      source_grad_r = (label[0]==True) * (source_k[i, j] - source_k[i, j+1])
      source_grad_l = (label[1]==True) * (source_k[i, j] - source_k[i, j-1])
      source_grad_b = (label[2]==True) * (source_k[i, j] - source_k[i+1, j])
      source_grad_u = (label[3]==True) * (source_k[i, j] - source_k[i-1, j])
      ## laplace for target image
      target_grad_r = (label[0]==True) * (target_k[i, j] - target_k[i, j+1])
      target_grad_l = (label[1]==True) * (target_k[i, j] - target_k[i, j-1])
      target_grad_b = (label[2]==True) * (target_k[i, j] - target_k[i+1, j])
      target_grad_u = (label[3]==True) * (target_k[i, j] - target_k[i-1, j])

      source_grad_value = []
      source_grad_value.append(source_grad_r if (abs(source_grad_r)>abs(target_grad_r)) else target_grad_r)
      source_grad_value.append(source_grad_l if (abs(source_grad_l)>abs(target_grad_l)) else target_grad_l)
      source_grad_value.append(source_grad_b if (abs(source_grad_b)>abs(target_grad_b)) else target_grad_b)
      source_grad_value.append(source_grad_u if (abs(source_grad_u)>abs(target_grad_u)) else target_grad_u)
      ## consider the dirichlet boundary condition
      if (inner_boundary[i][j])==1:
        dirichlet_condition_value = (label[0]==False)*target_k[i,j+1]+(label[1]==False)*target_k[i,j-1]\
          +(label[2]==False)*target_k[i+1,j]+(label[3]==False)*target_k[i-1,j]
      else:
        dirichlet_condition_value = 0
      
      u[k,index] = sum(source_grad_value) + dirichlet_condition_value
    
  return u


def texture_flatten(source, target, omega, inner_boundary, has_neighbor):
  u = np.zeros((3,omega.shape[0]))
  gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
  edge_mask = edge_detector(np.array(gray*255, dtype=np.uint8),10)

  for k in range(3):
    for index in range(omega.shape[0]):
      i, j = omega[index]
      label = has_neighbor[index]
      source_k = source[:,:,k]
      target_k = target[:,:,k]
      ## laplace for source image
      source_grad_value = (label[0]==True)*(source_k[i,j]-source_k[i,j+1])*(edge_mask[i][j])+\
        (label[1]==True)*(source_k[i,j]-source_k[i,j-1])*(edge_mask[i][j-1])+\
      (label[2]==True)*(source_k[i,j]-source_k[i+1,j])*(edge_mask[i][j])+\
        (label[3]==True)*(source_k[i,j]-source_k[i-1,j])*(edge_mask[i-1][j])

  ## consider the dirichlet boundary condition
      if (inner_boundary[i][j])==1:
        dirichlet_condition_value = (label[0]==False)*target_k[i,j+1]+(label[1]==False)*target_k[i,j-1]\
          +(label[2]==False)*target_k[i+1,j]+(label[3]==False)*target_k[i-1,j]
      else:
        dirichlet_condition_value = 0
      
      u[k,index] = source_grad_value + dirichlet_condition_value

  return u


def check_neighbors(mask, y, x):

  h, w = mask.shape

  if((y>=0) and (y<=h-1) and (x>=0) and (x<=w-1)):
    if(mask[y][x]==1):
      return True
    else:
      return False
  else:
    return False


def get_coeff_matrix(omega_list, has_neighbor, omega_yx):

  ## get the coefficient matrix A

  ## create empty sparse matrix
  num = omega_list.shape[0]
  A = sp.lil_matrix((num,num), dtype=np.float32)
  dic = {0:[0,1],1:[0,-1],2:[1,0],3:[-1,0]}
  
  for i in range(num):
    ## fill 4 or -1
    ## center
    A[i, i] = 4
    y, x = omega_list[i]

    for k in range(4):
      if(has_neighbor[i][k]):
        j = omega_yx[y+dic[k][0]][x+dic[k][1]]
        A[i,j] = -1

  return A


def edge_detector(gray, weight):
  ## implement the edge detector for texture flattening, use weight to change the thickness of edge.

  ### get edge from filter
  original_edge = cv2.Canny(gray, 100, 200)
  edge = np.zeros((original_edge.shape[0], original_edge.shape[1]), dtype=np.uint8)

  ### make edge bold by using weight source_grad_value
  for i in range(edge.shape[0]):
    for j in range(edge.shape[1]):
        if(original_edge[i][j] != 0):
          for m in range(-weight, weight):
            for n in range(-weight, weight):
              edge_i = min(max(0,i+m), edge.shape[0]-1)
              edge_j = min(max(0,j+n), edge.shape[1]-1)
              edge[edge_i][edge_j] = 1

  return edge


def indices(mask):
  #To get whether a pixel has neighbor pixel and a list for omega's index

  ## height and width of mask
  h, w = mask.shape

  ## get the coordinates that the elements are not 0 in mask, which is the omega area.
  omega = np.nonzero(mask)
  x = np.reshape(omega[1],[omega[1].shape[0],1])
  y = np.reshape(omega[0],[omega[0].shape[0],1])
  omega_list = np.concatenate([y,x],1)

  has_neighbor = []
  omega_yx = np.zeros((h,w),dtype=np.int32)
  for index in range(omega_list.shape[0]):

    ## pixel location
    i, j = omega_list[index]

    ## check whether there are neighbors (right, left, down, up)
    has_neighbor.append([check_neighbors(mask,i,j+1),check_neighbors(mask,i,j-1),check_neighbors(mask,i+1,j),check_neighbors(mask,i-1,j)])

    ## store index to list
    omega_yx[i][j] = index

  return omega_list, np.array(has_neighbor,dtype=bool), omega_yx


def poissonImageEditing(src, mask, tar, method):
  ### create inner_boundary mask
  inner_boundary = extract_inner_boundary(mask) # uint8
  mask = np.array(mask, dtype=np.uint8)
  ### get omega, neighborhoods flag
  omega, has_neighbor, yx_omega = indices(mask)
  print("start the operation\n")
  ### fill A
  print("step1: filling coefficient matrix: A")
  A = get_coeff_matrix(omega, has_neighbor, yx_omega)

  ### fill u for each color channel
  print("step2: filling gradient matrix: b")
  u = np.zeros((3,omega.shape[0]))
  u_b = np.zeros(omega.shape[0])
  u_g = np.zeros(omega.shape[0])
  u_r = np.zeros(omega.shape[0])
  ## select process type
  if(method == "ig"): ## importing laplace
    u = importing_laplace(src, tar, omega, inner_boundary, has_neighbor)
    u_b = u[0,:]
    u_g = u[1,:]
    u_r = u[2,:]
  if(method == "mg"): ## mixing laplace
    u =  mixing_laplace(src, tar, omega, inner_boundary, has_neighbor)
    u_b = u[0,:]
    u_g = u[1,:]
    u_r = u[2,:]
  if(method == "tf"): ## texture flattening
    u = texture_flatten(src, tar, omega, inner_boundary, has_neighbor)
    u_b = u[0,:]
    u_g = u[1,:]
    u_r = u[2,:]

  ### solve
  print("step3: solve Au = b\n")
  x_b, info_b = sp.linalg.cg(A, u_b)
  x_g, info_g = sp.linalg.cg(A, u_g)
  x_r, info_r = sp.linalg.cg(A, u_r)
  print("The operation is finishes!\n")

  ### create output by using x
  seamless_cloning = tar.copy()
  cloning = tar.copy()

  for index in range(omega.shape[0]):

    i, j = omega[index]
  
    ## normal
    seamless_cloning[i][j][0] = np.clip(x_b[index], 0.0, 1.0)
    seamless_cloning[i][j][1] = np.clip(x_g[index], 0.0, 1.0)
    seamless_cloning[i][j][2] = np.clip(x_r[index], 0.0, 1.0)

    ## cloning
    cloning[i][j][0] = src[i][j][0]
    cloning[i][j][1] = src[i][j][1]
    cloning[i][j][2] = src[i][j][2]


  return (np.array(seamless_cloning*255, dtype=np.uint8), 
          np.array(cloning*255, dtype=np.uint8))