#!/usr/bin/env python
# coding: utf-8


import cv2
# from google.colab.patches import cv2_imshow
import numpy as np
import math
from PIL import Image
import os
import json
import pandas as pd
from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.ops import unary_union
from numba import cuda, jit, njit
import pkg_resources


def get_demo_video_data():
    video_path = pkg_resources.resource_filename(__name__, "panoMosaics/demo_data/2023.03.29-17.39.21-main.avi")
    main_frame_arr = video_to_frame_arr(video_path)
    return main_frame_arr

def get_demo_detic_data():
    data_path = pkg_resources.resource_filename(__name__, "panoMosaics/demo_data/2023.03.29-17.39.21-detic:image.json")
    with open(data_path, 'r') as j:
     detic_dict = json.loads(j.read())
    return detic_dict

def get_sfs_arr(main_frame_arr):
  frame_sfs_arr = []
  sfs = 0
  for i in range(len(main_frame_arr)):
    frame_sfs_arr.append(sfs)
    sfs = sfs + (1/15)
  return frame_sfs_arr

def get_timestamp_arr(detic_dict):
  timestamps_sfs_arr = []
  start_timestamp = detic_dict[0]["timestamp"]
  for i in range(len(detic_dict)):
    current_timestamp = detic_dict[i]["timestamp"]
    sec_from_start = get_sec_from_start(current_timestamp, start_timestamp)
    timestamps_sfs_arr.append(sec_from_start)
  return timestamps_sfs_arr

def get_frame_timestamps_arr(main_frame_arr, detic_dict, frame_sfs_arr, timestamps_sfs_arr):
  frames_timestamps_arr = []
  arr = timestamps_sfs_arr
  n = len(arr)
  for i in range(len(main_frame_arr)):
    target = frame_sfs_arr[i]
    closest = findClosest(arr, n, target)
    target_idx = timestamps_sfs_arr.index(closest)
    timestamp = detic_dict[target_idx]["timestamp"]
    frames_timestamps_arr.append(timestamp)
  return frames_timestamps_arr

#video from camera starts with 760 width 428 height
def video_to_frame_arr(video_path):
  cap = cv2.VideoCapture(video_path)
  frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  max_frame = frame_count
  output_arr = []
  # naive version (took 24s just appending frames)
  success, img = cap.read()
  while max_frame >= 0:
    max_frame = max_frame - 1
    output_arr.append(img)
    # read next frame
    success, img = cap.read()
  return output_arr


def get_keypoints_descriptors(image):
  # Reading the image and converting into B/W
  #image = cv2.imread(image_path)
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Applying the function
  sift = cv2.xfeatures2d.SIFT_create()
  kp, des = sift.detectAndCompute(gray_image, None)
    
  # uncomment to draw keypoints on image
  #kp_image = cv2.drawKeypoints(image, kp, None, color=(
      #0, 255, 0)
      #, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  
  return(kp, des)




def feature_matching(des, des2):
  FLANN_INDEX_KDTREE = 1
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)
  flann = cv2.FlannBasedMatcher(index_params, search_params)
  matches = flann.knnMatch(des,des2,k=2)
  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)
  return good



def get_homography_matrix(image, image2, kp, kp2, good, MIN_MATCH_COUNT):
  if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      matchesMask = mask.ravel().tolist()
      h = image.shape[0]
      w = image.shape[1]
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      dst = cv2.perspectiveTransform(pts,M)
      #image2_lines = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
  else:
      print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
      matchesMask = None
      M = []
  return M
#uncomment this if you want to circle the RANSAC inliers/outliers and how they connect between the images
"""
  draw_params = dict(matchColor = (0,255,0), # draw matches in green 
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = cv2.DrawMatchesFlags_DEFAULT) #flags was 2
  image3 = cv2.drawMatches(image,kp,image2,kp2,good,None,**draw_params)
"""


#this is the same as get_homography_matrix but it also returns a side by side of the two images with RANSAC inliers/outliers circled and the inliers connected between images
def get_homography_matrix_old(image, image2, kp, kp2, good, MIN_MATCH_COUNT):
  if len(good)>MIN_MATCH_COUNT:
      src_pts = np.float32([ kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
      dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
      M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
      matchesMask = mask.ravel().tolist()
      h = image.shape[0]
      w = image.shape[1]
      pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
      dst = cv2.perspectiveTransform(pts,M)
      #image2_lines = cv2.polylines(image2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
  else:
      print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
      matchesMask = None
      M = []
  draw_params = dict(matchColor = (0,255,0), # draw matches in green 
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = cv2.DrawMatchesFlags_DEFAULT) #flags was 2
  image3 = cv2.drawMatches(image,kp,image2,kp2,good,None,**draw_params)
  return M, image3


def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homography matrix H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result



#save array of frames to video 
def save_to_video(output_frame_arr, fps, output_path):
  height,width,layers=output_frame_arr[0].shape
  video=cv2.VideoWriter(filename = output_path,fourcc = 0x7634706d,fps = fps,frameSize = (width, height))
  for j in range(len(output_frame_arr)):
    video.write(output_frame_arr[j])
  video.release()



#resize all frames in an array to the same resolution (specify desired width and height as parameters)
def resize_all(pano_frames_arr, width, height):
  resized_pano_arr = []
  height,width,layers=pano_frames_arr[0].shape
  for i in range(len(pano_frames_arr)):
    resized_pano_arr.append(cv2.resize(pano_frames_arr[i], (width, height)))
  return resized_pano_arr


def classic_bounding_boxes(main_frame_arr, detic_dict, frames_timestamps_arr, index_range):
  color = (0, 0, 255)
  thickness = 2
  output_frames = []
  for j in range(len(main_frame_arr)):  # for each frame
    image = main_frame_arr[j].copy()
    index = next((i for i, obj in enumerate(detic_dict) if obj['timestamp'] == frames_timestamps_arr[index_range[0] + j]), -1)
    for i in range(len(detic_dict[index]["values"])):  # for each detected object
      x = int(detic_dict[index]["values"][i]["xyxyn"][0] * 760)
      w = int(detic_dict[index]["values"][i]["xyxyn"][2] * 760)
      y = int(detic_dict[index]["values"][i]["xyxyn"][1] * 428)
      h = int(detic_dict[index]["values"][i]["xyxyn"][3] * 428)
      start_point = (x, y)
      end_point = (w, h)

      # draw object bounding box
      image = cv2.rectangle(image, start_point, end_point, color, thickness)

      # Draw red background rectangle
      image = cv2.rectangle(image, (x, y-15), (x + (w - x), y), (0,0,255), -1)  # TODO: figure this out

      # Add text
      image = cv2.putText(image, detic_dict[index]["values"][i]["label"], (x + 2,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    output_frames.append(image)
    
  return output_frames


#given two frames, this function warps them to the same plane and translates them to their proper position on a shared background by adding padding where needed
#You need to use this if you don't want the final panorama to be cropped to the resolution of one of the original frames
def warpPerspectivePadded(src, dst, transf):

    src_h, src_w = src.shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

    trans_lin_homg_pts = transf.dot(lin_homg_pts)
    trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

    minX = np.min(trans_lin_homg_pts[0,:])
    minY = np.min(trans_lin_homg_pts[1,:])
    maxX = np.max(trans_lin_homg_pts[0,:])
    maxY = np.max(trans_lin_homg_pts[1,:])

    # calculate the needed padding and create a blank image to place dst within
    dst_sz = list(dst.shape)
    pad_sz = dst_sz.copy() # to get the same number of channels
    pad_sz[0] = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
    pad_sz[1] = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)
    dst_pad = np.zeros(pad_sz, dtype=np.uint8)

    # add translation to the transformation matrix to shift to positive values
    anchorX, anchorY = 0, 0
    transl_transf = np.eye(3,3)
    if minX < 0: 
        anchorX = np.round(-minX).astype(int)
        transl_transf[0,2] += anchorX
    if minY < 0:
        anchorY = np.round(-minY).astype(int)
        transl_transf[1,2] += anchorY
    new_transf = transl_transf.dot(transf)
    new_transf /= new_transf[2,2]
    
    dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst
    dest_pad_pre_warp = dst_pad

    warped = cv2.warpPerspective(src, new_transf, (pad_sz[1],pad_sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return dst_pad, warped, dest_pad_pre_warp




#warps a single point from one plane to another given a homography matrix M
def warp_point(x, y, M):
    d = M[2][0] * x + M[2][1] * y + M[2][2]

    return (
        int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d), # x
        int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d), # y
    )




#this is like warpPerspectivePadded but for N frames instead of 2
def warp_n_with_padding(dst, src_list, transf_list, main_frame_arr):
  #main_frame_arr = main_frame_arr2
  #dst = main_frame_arr[505]
  #src_list = [main_frame_arr[450], main_frame_arr[480], main_frame_arr[508], main_frame_arr[512], main_frame_arr[525]]
  #transf_list = [hm13, hm23, hm43, hm53, hm63]

  pad_sz_0_arr = []
  pad_sz_1_arr = []
  minMaxXY_arr = []
  dst_sz = list(dst.shape)

  for i in range(len(src_list)):
    src_h, src_w = src_list[i].shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])
    trans_lin_homg_pts = transf_list[i].dot(lin_homg_pts)
    trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

    minX = np.min(trans_lin_homg_pts[0,:])
    minY = np.min(trans_lin_homg_pts[1,:])
    maxX = np.max(trans_lin_homg_pts[0,:])
    maxY = np.max(trans_lin_homg_pts[1,:])

    pad_sz0 = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
    pad_sz1 = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)

    minMaxXY_arr.append([minX, minY, maxX, maxY])
    pad_sz_0_arr.append(pad_sz0)
    pad_sz_1_arr.append(pad_sz1)

  # calculate the needed padding and create a blank image to place dst within
  pad_sz = dst_sz.copy() # to get the same number of channels
  pad_sz[0] = max(pad_sz_0_arr)
  pad_sz[1] = max(pad_sz_1_arr)
  indexY = pad_sz_0_arr.index(pad_sz[0])
  indexX = pad_sz_1_arr.index(pad_sz[1])
  minY = minMaxXY_arr[indexY][1]
  maxY = minMaxXY_arr[indexY][3]
  minX = minMaxXY_arr[indexX][0]
  maxX = minMaxXY_arr[indexX][2]
  dst_pad = np.zeros(pad_sz, dtype=np.uint8)

  #add translation to ALL transformation matrices to shift to positive values
  new_transf_list = []
  anchorX_list = []
  anchorY_list = []
  for i in range(len(transf_list)):
    anchorX, anchorY = 0, 0
    transl_transf = np.eye(3,3)
    if minX < 0: 
        anchorX = np.round(-minX).astype(int)
        transl_transf[0,2] += anchorX
    if minY < 0:
        anchorY = np.round(-minY).astype(int)
        transl_transf[1,2] += anchorY
    new_transf = transl_transf.dot(transf_list[i])
    new_transf /= new_transf[2,2]
    new_transf_list.append(new_transf)
    anchorX_list.append(anchorX)
    anchorY_list.append(anchorY)

  anchorX = max(anchorX_list)
  anchorY = max(anchorY_list)
  dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst

  warped_src_arr = []
  for i in range(len(src_list)):
    warped = cv2.warpPerspective(src_list[i], new_transf_list[i], (pad_sz[1],pad_sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_src_arr.append(warped)
  
  return dst_pad, warped_src_arr, new_transf_list, anchorX, anchorY



#converts all the warped + translated pieces of the panorama from RGB to RGBA images (you'll need the alpha channel [which sets opacity] for compositing)
def get_rgba_im_arr(dst_pad, warped_src_arr):  
  im_arr = []
  im = Image.fromarray(dst_pad)
  im = im.convert("RGBA")
  im = np.asarray(im)
  im_arr.append(im)
  for i in range(len(warped_src_arr)):
    im2 = Image.fromarray(warped_src_arr[i])
    im2 = im2.convert("RGBA")
    im2 = np.asarray(im2)
    im_arr.append(im2)
  
  return im_arr



#converts an array containing an RGB image to an array containing an RGBA image (adds alpha channel)
def rgba_to_rgb(comp_arr):
  im = Image.fromarray((comp_arr).astype(np.uint8))
  im = im.convert('RGB')
  im = np.asarray(im)
  return im


#converts an array containing an RGB image to an array containing an RGBA image (adds alpha channel)
def rgb_to_rgba(im):
  im = Image.fromarray((im).astype(np.uint8))
  im = im.convert('RGBA')
  im = np.asarray(im)
  return im




#this function is like. the slowest possible way to do this. Will be updating soon
def alpha_composite_n_images(im_arr):
  #naive solution
  comp = []
  im = im_arr[0]
  for row in range(im.shape[0]):
    comp_inner = []
    for col in range(im.shape[1]):
      #figure out which images are black at this pixel
      not_black_list = []
      black_list = []
      for i in range(len(im_arr)):
        if im_arr[i][row][col][0] == 0 and im_arr[i][row][col][1] == 0 and im_arr[i][row][col][2] == 0:
          black_list.append(im_arr[i])
        else:
          not_black_list.append(im_arr[i])
      #if all images are black, set to transparent
      if len(not_black_list) == 0:
        comp_inner.append([0, 0, 0, 0])
      #if only one image is NOT black, use it
      elif len(not_black_list) == 1:
        comp_inner.append(not_black_list[0][row][col])
      #if multiple images are not black, alpha blend them all together
      else:
        alpha = 1/len(not_black_list)
        channel1 = 0
        channel2 = 0
        channel3 = 0
        for j in range(len(not_black_list)):
            channel1 = channel1 + alpha * not_black_list[j][row][col][0]
            channel2 = channel2 + alpha * not_black_list[j][row][col][1]
            channel3 = channel3 + alpha * not_black_list[j][row][col][2]
        comp_inner.append([channel1, channel2, channel3, 255])
    comp.append(comp_inner)
  comp_arr = np.array(comp)
  return comp_arr




@njit(parallel=True)
def alpha_composite_n_images_parallel(im_arr):
    im = im_arr[0]
    comp = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.float32)

    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            #figure out which images are black at this pixel
            not_black_list = []
            black_list = []
            for i in range(len(im_arr)):
                if im_arr[i][row][col][0] == 0 and im_arr[i][row][col][1] == 0 and im_arr[i][row][col][2] == 0:
                    black_list.append(im_arr[i])
                else:
                    not_black_list.append(im_arr[i])
            #if all images are black, set to transparent
            if len(not_black_list) == 0:
                comp[row][col][0] = 0
                comp[row][col][1] = 0
                comp[row][col][2] = 0
                comp[row][col][3] = 0
            #if only one image is NOT black, use it
            elif len(not_black_list) == 1:
                comp[row][col] = not_black_list[0][row][col]
            #if multiple images are not black, alpha blend them all together
            else:
                alpha = 1/len(not_black_list)
                channel1 = 0
                channel2 = 0
                channel3 = 0
                for j in range(len(not_black_list)):
                    channel1 = channel1 + alpha * not_black_list[j][row][col][0]
                    channel2 = channel2 + alpha * not_black_list[j][row][col][1]
                    channel3 = channel3 + alpha * not_black_list[j][row][col][2]
                comp[row][col][0] = channel1
                comp[row][col][1] = channel2
                comp[row][col][2] = channel3
                comp[row][col][3] = 255
    return comp



def alpha_composite_two(im, im2):
  comp = []
  #np.zeros((im.shape[0], im.shape[1], im.shape[2]))
  alpha = 0.5
  for row in range(im.shape[0]):
    comp_inner = []
    for col in range(im.shape[1]):
      #if one image is black, just use the other
      if (im[row][col][0] == 0 and im[row][col][1] == 0 and im[row][col][2] == 0) and (im2[row][col][0] != 0 or im2[row][col][1] != 0 or im2[row][col][2] != 0):
        comp_inner.append(im2[row][col])
      elif (im2[row][col][0] == 0 and im2[row][col][1] == 0 and im2[row][col][2] == 0) and (im[row][col][0] != 0 or im[row][col][1] != 0 or im[row][col][2] != 0):
        comp_inner.append(im[row][col])
      #if both images are black, set to transparent
      elif (im[row][col][0] == 0 and im[row][col][1] == 0 and im[row][col][2] == 0) and (im2[row][col][0] == 0 and im2[row][col][1] == 0 and im2[row][col][2] == 0):
        comp_inner.append([0, 0, 0, 0])
      #if both pixels are not black, alpha blend
      else:
        channel1 = alpha * im[row][col][0] + (1 - alpha) * im2[row][col][0]
        channel2 = alpha * im[row][col][1] + (1 - alpha) * im2[row][col][1]
        channel3 = alpha * im[row][col][2] + (1 - alpha) * im2[row][col][2]
        comp_inner.append([channel1, channel2, channel3, 255])
    comp.append(comp_inner)
  return comp



#warps panorama back to rectangle after compositing. Doesn't work for every case yet
#this version works by tracing where the corners of the original frames are warped to in the panorama and using those to "pull" it back into a rectangle
def warp_back_to_rect_up(og_src, org_dst, final_width, final_height, anchorX, anchorY, hm_og_src_og_dst, comp_arr):
  og_dst_width = org_dst.shape[1]
  og_dst_height =org_dst.shape[0]
  og_dst_corners = [[anchorX, anchorY], [anchorX + og_dst_width, anchorY], [anchorX + og_dst_width, anchorY + og_dst_height], [anchorX, anchorY + og_dst_height]] #clockwise from top left

  og_src_width = og_src.shape[1]
  og_src_height = og_src.shape[0]
  og_src_warped_top_left = warp_point(0, 0, hm_og_src_og_dst)
  og_src_warped_top_right = warp_point(og_src_width, 0, hm_og_src_og_dst)
  og_src_warped_bottom_right = warp_point(og_src_width, og_src_height, hm_og_src_og_dst)
  og_src_warped_bottom_left = warp_point(0, og_src_height, hm_og_src_og_dst)
  og_src_warped_corners = [og_src_warped_top_left, og_src_warped_top_right, og_src_warped_bottom_right, og_src_warped_bottom_left]

  src_quad_list = np.float32([og_dst_corners[0], og_dst_corners[1], og_src_warped_corners[2], og_src_warped_corners[3]])
  dst_quad_list = np.float32([[0, 0], [final_width, 0], [final_width, final_height], [0, final_height]])

  homography_matrix = cv2.getPerspectiveTransform(src_quad_list, dst_quad_list)

  rect = cv2.warpPerspective(comp_arr, homography_matrix, (760, 428))

  for j in range(rect.shape[0] - 1):
    for i in range(len(rect[0])):
      if rect[j][i][3] == 255:
        top_left_x = i
        break

  for j in range(rect.shape[0] - 1):
    for i in range(len(rect[0])):
      if rect[j][(len(rect[0]) - 1) - i][3] == 255:
        top_right_x = i
        break

  for j in range(rect.shape[0] - 1):
    for i in range(rect.shape[1] - 1):
      if rect[(rect.shape[0] - 1) - j][i][3] == 255:
        bottom_left_x = i
        break

  for j in range(rect.shape[0] - 1):
    for i in range(len(rect[rect.shape[0] - 1])):
      if rect[(rect.shape[0] - 1) - j][(rect.shape[0] - 1) - i][3] == 255:
        bottom_right_x = i
        break

  for j in range(rect.shape[1] - 1):
    for i in range(len(rect[:,0])):
      if rect[i][j][3] == 255:
        top_left_y = i
        break

  for j in range(rect.shape[1] - 1):
    for i in range(rect.shape[0]):
      if rect[i][(rect.shape[1] - 1) - j][3] == 255:
        top_right_y = i
        break

  for j in range(rect.shape[1] - 1):
    for i in range(rect.shape[0]):
      if rect[(rect.shape[0] - 1) - i][(rect.shape[1] - 1) - j][3] == 255:
        bottom_right_y = i
        break

  for j in range(rect.shape[1] - 1):
    for i in range(len(rect[:,0])):
      if rect[(rect.shape[0] - 1) - i][j][3] == 255:
        bottom_left_y = i
        break

  #crop rectangle using where src_quad_list warped to as 4 corners 
  top_bound = max(top_left_y, top_right_y)
  bottom_bound = rect.shape[0] - min(bottom_left_y, bottom_right_y)
  left_bound = max(top_left_x, bottom_left_x)
  right_bound = rect.shape[1] - min(top_right_x, bottom_right_x)

  del_row_arr = []
  #rows/y
  for i in range(rect.shape[0]):
    if i < top_bound or i > bottom_bound:
      del_row_arr.append(i)
  rect = np.delete(rect, del_row_arr, 0)

  del_col_arr = []
  #cols/x
  for j in range(rect.shape[1]):
    if j < left_bound or j > right_bound:
      del_col_arr.append(j)
  rect = np.delete(rect, del_col_arr, 1)

  rect = cv2.resize(rect, (760, 428))

  return rect, homography_matrix




#crops any fully transparent rows or columns off an image
def crop_transparent(comp_arr):
  col_sums = np.sum(comp_arr, axis = 0)
  del_cols_list = np.where(col_sums == 0)[0]
  comp_arr = np.delete(comp_arr, del_cols_list, 1)
  row_sums = np.sum(comp_arr, axis = 1)
  del_rows_list = np.where(row_sums == 0)[0]
  comp_arr = np.delete(comp_arr, del_rows_list, 0)
  return comp_arr



#this is a helper function used for matching the video frames to object detection outputs
# find element closest to given target using binary search.
 
def findClosest(arr, n, target):
 
    # Corner cases
    if (target <= arr[0]):
        return arr[0]
    if (target >= arr[n - 1]):
        return arr[n - 1]
 
    # Doing binary search
    i = 0; j = n; mid = 0
    while (i < j):
        mid = (i + j) // 2
 
        if (arr[mid] == target):
            return arr[mid]
 
        # If target is less than array
        # element, then search in left
        if (target < arr[mid]) :
 
            # If target is greater than previous
            # to mid, return closest of two
            if (mid > 0 and target > arr[mid - 1]):
                return getClosest(arr[mid - 1], arr[mid], target)
 
            # Repeat for left half
            j = mid
         
        # If target is greater than mid
        else :
            if (mid < n - 1 and target < arr[mid + 1]):
                return getClosest(arr[mid], arr[mid + 1], target)
                 
            # update i
            i = mid + 1
         
    # Only single element left after search
    return arr[mid]
 
 
# Method to compare which one is the more close.
# We find the closest by taking the difference
# between the target and both values. It assumes
# that val2 is greater than val1 and target lies
# between these two.
def getClosest(val1, val2, target):
 
    if (target - val1 >= val2 - target):
        return val2
    else:
        return val1



def get_sec_from_start(current_timestamp, start_timestamp):
  sec_from_start = (float(current_timestamp[0:-2]) - float(start_timestamp[0:-2]))/1000
  return sec_from_start


def warp_one_bbox(x, y, w, h, M):
  
  #warp top left corner 
  new_x, new_y = warp_point(x, y, M)

  #warp bottom right corner
  new_w, new_h = warp_point(w, h, M)

  return new_x, new_y, new_w, new_h


#gets the x,y coords of how much the destination image was translated during the warping process
def get_anchors(dst, src_list, transf_list, main_frame_arr):
  pad_sz_0_arr = []
  pad_sz_1_arr = []
  minMaxXY_arr = []
  dst_sz = list(dst.shape)

  for i in range(len(src_list)):
    src_h, src_w = src_list[i].shape[:2]
    lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])
    trans_lin_homg_pts = transf_list[i].dot(lin_homg_pts)
    trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

    minX = np.min(trans_lin_homg_pts[0,:])
    minY = np.min(trans_lin_homg_pts[1,:])
    maxX = np.max(trans_lin_homg_pts[0,:])
    maxY = np.max(trans_lin_homg_pts[1,:])

    pad_sz0 = np.round(np.maximum(dst_sz[0], maxY) - np.minimum(0, minY)).astype(int)
    pad_sz1 = np.round(np.maximum(dst_sz[1], maxX) - np.minimum(0, minX)).astype(int)

    minMaxXY_arr.append([minX, minY, maxX, maxY])
    pad_sz_0_arr.append(pad_sz0)
    pad_sz_1_arr.append(pad_sz1)

  # calculate the needed padding and create a blank image to place dst within
  pad_sz = dst_sz.copy() # to get the same number of channels
  pad_sz[0] = max(pad_sz_0_arr)
  pad_sz[1] = max(pad_sz_1_arr)
  indexY = pad_sz_0_arr.index(pad_sz[0])
  indexX = pad_sz_1_arr.index(pad_sz[1])
  minY = minMaxXY_arr[indexY][1]
  maxY = minMaxXY_arr[indexY][3]
  minX = minMaxXY_arr[indexX][0]
  maxX = minMaxXY_arr[indexX][2]
  #dst_pad = np.zeros(pad_sz, dtype=np.uint8)

  #add translation to ALL transformation matrices to shift to positive values
  new_transf_list = []
  anchorX_list = []
  anchorY_list = []
  for i in range(len(transf_list)):
    anchorX, anchorY = 0, 0
    transl_transf = np.eye(3,3)
    if minX < 0: 
        anchorX = np.round(-minX).astype(int)
        transl_transf[0,2] += anchorX
    if minY < 0:
        anchorY = np.round(-minY).astype(int)
        transl_transf[1,2] += anchorY
    new_transf = transl_transf.dot(transf_list[i])
    new_transf /= new_transf[2,2]
    new_transf_list.append(new_transf)
    anchorX_list.append(anchorX)
    anchorY_list.append(anchorY)

  anchorX = max(anchorX_list)
  anchorY = max(anchorY_list)
  #dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst

  #warped_src_arr = []
  #for i in range(len(src_list)):
    #warped = cv2.warpPerspective(src_list[i], new_transf_list[i], (pad_sz[1],pad_sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    #warped_src_arr.append(warped)
  
  return anchorX, anchorY


#input: an array of frames and a list of indices to stitch (can be all)
#outputs: an array of panoramas (on a fixed-size background), with each consecutive panorama adding the next frame from the input list of indices
#this also writes all of the output frames to a folder so they can easily be saved/reloaded 

@jit()
def stitch_one_at_a_time(indices, main_frame_arr, background_width, background_height): #I've been using background_width = 2280 and background_height = 1284 (3 times the width and height of one frame) as the default
  final_pano_frames = []
  output_min_x = 0
  output_max_x = 0
  output_min_y = 0
  output_max_y = 0
  
  for f in range(1, len(indices)):
  
    if len(final_pano_frames) == 0: #there's some extra steps for the first one
      src = rgb_to_rgba(main_frame_arr[indices[1]])
      dst = rgb_to_rgba(main_frame_arr[indices[0]])

      dst_sz = list(dst.shape)
      pad_sz = [dst.shape[0] * 3, dst.shape[1] * 3, 4]
      anchorX = dst.shape[1]-1 #this just places the first frame roughly in the center of the background (when the background is set to default values mentioned in above comment)
      anchorY = dst.shape[0]-1

      output_min_x = anchorX
      output_max_x = anchorX + dst_sz[1]-1
      output_min_y = anchorY
      output_max_y = anchorY + dst_sz[0]-1

      dst_pad = np.zeros(pad_sz, dtype=np.uint8)
      dst_pad[anchorY:anchorY+dst_sz[0], anchorX:anchorX+dst_sz[1]] = dst

      #warp src into correct place
      kp_src, des_src = get_keypoints_descriptors(src)
      kp_dst, des_dst = get_keypoints_descriptors(dst)
      matches = feature_matching(des_src, des_dst)
      transf = get_homography_matrix(src, dst, kp_src, kp_dst, matches, 4)

      src_h, src_w = src.shape[:2]
      lin_homg_pts = np.array([[0, src_w, src_w, 0], [0, 0, src_h, src_h], [1, 1, 1, 1]])

      trans_lin_homg_pts = transf.dot(lin_homg_pts)
      trans_lin_homg_pts /= trans_lin_homg_pts[2,:]

      # add translation to the transformation matrix to shift to positive values
      transl_transf = np.eye(3,3)
      transl_transf[0,2] += anchorX
      transl_transf[1,2] += anchorY
      new_transf = transl_transf.dot(transf)
      new_transf /= new_transf[2,2]

      #take the corners of src and dst as polygons in shapely
      pts = np.float32([ [0,0],[src_w, 0],[src_w, src_h],[0, src_h] ]).reshape(-1,1,2)
      dst_pts = cv2.perspectiveTransform(pts,new_transf)

      src_polygon = Polygon([(dst_pts[0][0][0], dst_pts[0][0][1]), (dst_pts[1][0][0], dst_pts[1][0][1]), (dst_pts[2][0][0], dst_pts[2][0][1]), (dst_pts[3][0][0], dst_pts[3][0][1])])
      src_polygon_x,src_polygon_y = src_polygon.exterior.xy
      dst_polygon = Polygon([(anchorX,anchorY),(anchorX+760,anchorY),(anchorX+760,anchorY+428),(anchorX,anchorY+428)])

      #get overlap
      overlap = src_polygon.intersection(dst_polygon)
      overlap_x,overlap_y = overlap.exterior.xy
      
      #warp src
      warped_src = cv2.warpPerspective(src, new_transf, (pad_sz[1],pad_sz[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

      #use overlap_x, overlap_y to create "bounding box" around src polygon, jump to that bounding box and only check points inside of it for blending while compositing
      #compositing dst_pad and warped_src
      #keep dst_pad or warped_src values for pixels with only one or the other
      #if a pixel is inside the overlap polygon, blend dst_pad and warped_src with equal weights
      dst_pad_copy = dst_pad.copy()
      min_x = math.floor(min(src_polygon_x))
      max_x = math.floor(max(src_polygon_x))
      min_y = math.floor(min(src_polygon_y))
      max_y = math.floor(max(src_polygon_y))

      output_min_x = min(output_min_x, min_x)
      output_max_x = max(output_max_x, max_x)
      output_min_y = min(output_min_y, min_y)
      output_max_y = max(output_max_y, max_y)

      for i in range(max_y - min_y):
        for j in range(max_x - min_x):
          current_point = Point(min_x + j, min_y + i)
          if overlap.contains(current_point):
            dst_pad_copy[min_y + i][min_x + j] = [(dst_pad_copy[min_y + i][min_x + j][0] / 2 + warped_src[min_y + i][min_x + j][0] / 2), (dst_pad_copy[min_y + i][min_x + j][1] / 2 + warped_src[min_y + i][min_x + j][1] / 2), (dst_pad_copy[min_y + i][min_x + j][2] / 2 + warped_src[min_y + i][min_x + j][2] / 2), 255]
          elif src_polygon.contains(current_point):
            dst_pad_copy[min_y + i][min_x + j] = warped_src[min_y + i][min_x + j]
      
      #save frames
      final_pano_frames.append(dst_pad) #"pano" with only first frame/dst
      final_pano_frames.append(dst_pad_copy) #pano with first two frames (src and dst)
      cv2.imwrite("/content/final_pano_frames/" + str(0) + ".png", final_pano_frames[0][anchorY:anchorY+dst_sz[0]:1, anchorX:anchorX+dst_sz[1]:1])
      cv2.imwrite("/content/final_pano_frames/" + str(1) + ".png", final_pano_frames[1][output_min_y:output_max_y+1:1, output_min_x:output_max_x+1:1])


    else:
      prev_dst_pad_copy = dst_pad_copy.copy()
      src = rgb_to_rgba(main_frame_arr[indices[f]])
      dst = prev_dst_pad_copy

      #warp src into correct place
      kp_src, des_src = get_keypoints_descriptors(src)
      kp_dst, des_dst = get_keypoints_descriptors(dst)
      matches = feature_matching(des_src, des_dst)
      transf = get_homography_matrix(src, dst, kp_src, kp_dst, matches, 4)

      #take the corners of src and dst as polygons in shapely
      src_h, src_w = src.shape[:2]
      pts = np.float32([ [0,0],[src_w, 0],[src_w, src_h],[0, src_h] ]).reshape(-1,1,2)
      dst_pts = cv2.perspectiveTransform(pts,transf)

      polygons = [src_polygon, dst_polygon]
      dst_polygon = unary_union(polygons)

      src_polygon = Polygon([(dst_pts[0][0][0], dst_pts[0][0][1]), (dst_pts[1][0][0], dst_pts[1][0][1]), (dst_pts[2][0][0], dst_pts[2][0][1]), (dst_pts[3][0][0], dst_pts[3][0][1])])
      src_polygon_x,src_polygon_y = src_polygon.exterior.xy

      #get overlap
      overlap = src_polygon.intersection(dst_polygon)
      overlap_x,overlap_y = overlap.exterior.xy

      #warp src
      warped_src = cv2.warpPerspective(src, transf, (prev_dst_pad_copy.shape[1],prev_dst_pad_copy.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

      #compositing
      min_x = math.floor(min(src_polygon_x))
      max_x = math.floor(max(src_polygon_x))
      min_y = math.floor(min(src_polygon_y))
      max_y = math.floor(max(src_polygon_y))

      output_min_x = min(output_min_x, min_x)
      output_max_x = max(output_max_x, max_x)
      output_min_y = min(output_min_y, min_y)
      output_max_y = max(output_max_y, max_y)

      for i in range(max_y - min_y):
        for j in range(max_x - min_x):
          current_point = Point(min_x + j, min_y + i)
          if overlap.contains(current_point):
            prev_dst_pad_copy[min_y + i][min_x + j] = [(prev_dst_pad_copy[min_y + i][min_x + j][0] / 2 + warped_src[min_y + i][min_x + j][0] / 2), (prev_dst_pad_copy[min_y + i][min_x + j][1] / 2 + warped_src[min_y + i][min_x + j][1] / 2), (prev_dst_pad_copy[min_y + i][min_x + j][2] / 2 + warped_src[min_y + i][min_x + j][2] / 2), 255]
          elif src_polygon.contains(current_point):
            prev_dst_pad_copy[min_y + i][min_x + j] = warped_src[min_y + i][min_x + j]
      
      #save
      final_pano_frames.append(prev_dst_pad_copy)
      cv2.imwrite("/content/final_pano_frames/" + str(f) + ".png", final_pano_frames[f][output_min_y:output_max_y+1:1, output_min_x:output_max_x+1:1])

  return final_pano_frames



""" Bounding Box / time
- at every time step, draw all bounding boxes from the beginning of the vis to now
- rgba : a inversely proportional to (distance from the present)^2
-- play around with color, thickness & how much of the box is being drawn (whole box vs. picture-frame vs. just corners)
"""

def box_color(time_color, time_from_present, max_time=5):  # assumed seconds
    """ 
    input: box's time from the present moment, how far in the past boxes should show up
    output: [r, g, b, a] value of the box [0,255]
    """
    if time_from_present >= max_time:
        return [0,0,0,0]
    color_factor = (1 - time_from_present/max_time)**2  # get the squared dist from max time by normalizing [0,1]
    hex_color = time_color[round(color_factor * (len(time_color)-1))]
    return rgb_to_bgr(hex_to_rgb(hex_color)) + [round(color_factor*255)]  # add alpha scaler to return

def hex_to_rgb(hex_str):
    """ helper to convert rgb hex to its individual values in a list 
    ref: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python """
    h = hex_str.lstrip('#')
    return list(int(h[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_bgr(lst):  # pano works in bgr for some reason
    return [lst[2], lst[1], lst[0]]



""" Resources:
- highlighting an area - https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
- cv2 draw functions - https://docs.opencv.org/4.x/dc/da5/tutorial_py_drawing_functions.html
"""

def draw_side_box(image, x,y, w,h, color, thickness):
    line_len1 = (w-x)//8
    image = cv2.line(image, (x,y), (x+line_len1, y), color, thickness)
    image = cv2.line(image, (x,y), (x,h), color, thickness)
    image = cv2.line(image, (x,h), (x+line_len1, h), color, thickness) 
    return cv2.line(image, (x,(y+h)//2), (x,-line_len1+(y+h)//2), color, thickness)

def draw_center_point(image, x,y, w,h, color, thickness):
    rect_size =  3  # size of boxes that denote corners
    newx = (x+w)//2
    newy = (y+h)//2
    return cv2.rectangle(image, (newx,newy), (newx+rect_size, newy+rect_size), color, rect_size)

def draw_center_point_updated(image, x,y, w,h, color, thickness):
    newx = ((w-x)//2)+x
    newy = ((h-y)//2)+y
    return cv2.circle(image,(newx,newy), 5, color, -1)

def draw_box_corners(image, x,y, w,h, color, thickness):
    rect_size =  2  # size of boxes that denote corners
    image = cv2.rectangle(image, (x,y), (x+rect_size, y+rect_size), color, rect_size)
    image = cv2.rectangle(image, (w,y), (w+rect_size, y+rect_size), color, rect_size)
    image = cv2.rectangle(image, (x,h), (x+rect_size, h+rect_size), color, rect_size)
    return cv2.rectangle(image, (w,h), (w+rect_size, h+rect_size), color, rect_size)


# line_len = 15;
def draw_box_frame(image, x,y, w,h, color, thickness):
    line_len1 = (w-x)//8
    line_len2 = (y-h)//8
    # TODO: tweak positioning
    image = cv2.line(image, (x,y), (x+line_len1, y), color, thickness)
    image = cv2.line(image, (x,y), (x, y-line_len2), color, thickness)
    
    image = cv2.line(image, (w-line_len1,y), (w,y), color, thickness)
    image = cv2.line(image, (w,y), (w,y-line_len2), color, thickness)
    
    image = cv2.line(image, (x,h), (x+line_len1, h), color, thickness)
    image = cv2.line(image, (x,h+line_len2), (x, h), color, thickness)
    
    image = cv2.line(image, (w-line_len1,h), (w, h), color, thickness)
    return cv2.line(image, (w,h+line_len2), (w, h), color, thickness)
    
    
def draw_box(image, x,y, w,h, color, thickness):
    return cv2.rectangle(image, (x,y), (w,h), color, thickness)

def draw_line(image, x,y, w,h, color, thickness):
    # basic func to draw line from (x,y) to (w,h)
    return cv2.line(image, (x,y), (w,h), color, thickness)

def draw_arrow(image, x,y, w,h, color, thickness):
  tip_length = (1 / math.dist([x,y], [w,h])) * 20
  return cv2.arrowedLine(image, (x,y), (w,h), color, thickness, 5, 0, tip_length)

def get_closer_further(x, y, t): #determines which of points x and y is closer to t
  if math.dist(x, t) < math.dist(y, t):
    closer = x
    further = y
  else:
    closer = y
    further = x
  return closer, further

def get_closest(array_of_points, target):
  dist = 10000000000000000000000000000
  for i in range(len(array_of_points)):
    if math.dist(array_of_points[i], target) < dist:
      dist = math.dist(array_of_points[i], target)
      closest = array_of_points[i]
      closest_index = i
  array_of_points.pop(closest_index)
  return closest, array_of_points, closest_index

"""

INPUTS:
index_list: array of indices of the frames in main_frame_arr frames used to create panorama_image [NOTE: this function will assume all frames were warped to the plane of the frame at the first index in index_list (aka the first one is dst)]
frames_timestamps_arr: array of timestamps corresponding to each frame used to create panorama_image
detic_dict: object detection model outputs
panorama_image: the panorama you want to draw bounding boxes on
new_transf_list: an array of the homography matrices used to create panorama_image (should be returned by warp_n_with_padding)
anchorX: the X translation used to create panorama_image (should be returned by warp_n_with_padding)
anchorY: the Y translation used to create panorama_image (should be returned by warp_n_with_padding)
colors_list: array of the bounding box colors for each timestep
thickness: thickness of bounding boxes (2 seems to be a reasonable default)

OUTPUTS:
image with bounding boxes for all objects at from given indices 

"""
def draw_all_bounding_boxes_for_given_indices(index_list, frames_timestamps_arr, detic_dict, panorama_image,
                                              transf_index_dict, dist_index, anchorX, anchorY, colors_list, color_scheme, thickness, box_type='center_dot_lined_updated', object_subset={}):
    image = panorama_image.copy() #so it doesn't draw directly on the panorama in case the one without bounding boxes is needed later
    last_dot = {}
    obj_locations = {} #this is the dictionary tracks keeps a record of obj locations over all timesteps
    current_frame_obj_counts_all = []
    color_object_dict = {}
    for f in range(len(index_list)):
        current_frame_obj_count = {} #this is the dictionary tracks the number of duplicates within a single timestep
        index = next((i for i, obj in enumerate(detic_dict) if obj['timestamp'] == frames_timestamps_arr[index_list[f]]), -1)
        for i in range(len(detic_dict[index]["values"])):
            obj_name = detic_dict[index]["values"][i]["label"]
            if len(object_subset)>0 and obj_name not in object_subset:
                continue
            x = int(detic_dict[index]["values"][i]["xyxyn"][0] * 760)
            w = int(detic_dict[index]["values"][i]["xyxyn"][2] * 760)
            y = int(detic_dict[index]["values"][i]["xyxyn"][1] * 428)
            h = int(detic_dict[index]["values"][i]["xyxyn"][3] * 428)

            if index_list[f] == dist_index:  # dst image --> coord plane
                x = x + anchorX
                w = w + anchorX
                y = y + anchorY
                h = h + anchorY
            else:
                x = warp_point(x, y, transf_index_dict[index_list[f]])[0]
                y = warp_point(x, y, transf_index_dict[index_list[f]])[1]
                w = warp_point(w, h, transf_index_dict[index_list[f]])[0]
                h = warp_point(w, h, transf_index_dict[index_list[f]])[1]

            start_point = (x, y)
            end_point = (w, h)

            if color_scheme == 'time':
              #draw object bounding box (draw_box, draw_box_frame, draw_box_corners)
              new_color = box_color(colors_list, f, len(index_list))
              
              if box_type == 'box':
                  image = draw_box(image, x,y, w,h, new_color, thickness)
              elif box_type == 'frame':
                  image = draw_box_frame(image, x,y, w,h, new_color, thickness)
              elif box_type == 'corner_dot':
                  image = draw_box_corners(image, x,y, w,h, new_color, thickness)
              elif box_type == 'center_dot':
                  image = draw_center_point(image, x,y, w,h, new_color, thickness)
              elif box_type == 'center_dot_lined':
                  image = draw_center_point(image, x,y, w,h, new_color, thickness)
                  if obj_name in last_dot:
                      xnew = last_dot[obj_name][0]
                      ynew = last_dot[obj_name][1]
                      image = draw_line(image, (x+w)//2, (y+h)//2, xnew,ynew, new_color, thickness//2)
                  last_dot[obj_name] = [(x+w)//2, (y+h)//2]
              elif box_type == 'center_dot_lined_updated':
                image = draw_center_point_updated(image, x,y, w,h, new_color, thickness)
                #3 cases (in order addressed below, using duplicate to mean there are multiple in the same timestep): 
                  #1) duplicate object (so we've already seen it), 
                  #2) non-duplicate object we've already seen in a previous timestep, 
                  #3) object we're seeing for the first time ever
                if obj_name in current_frame_obj_count: #case 1
                  #update duplicate count
                  current_frame_obj_count[obj_name] = current_frame_obj_count[obj_name] + 1
                  duplicate_num = current_frame_obj_count[obj_name]
                  #if len(current_frame_obj_counts_all) >= 1 and current_frame_obj_counts_all[f-1][obj_name] > 1:  #case 1a
                  #else: #case 1b
                  if f == 0:
                    if duplicate_num == 2:
                      obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name)
                    obj_locations[obj_name + "_" + str(duplicate_num)] = [[(x+w)//2, (y+h)//2]]
                  else:
                    if duplicate_num == 2: #if this is the first duplicate for this object in this timestep (case 1b1)
                      current_loc = [(x+w)//2, (y+h)//2]
                      object_dict = obj_locations.pop(obj_name)
                      if len(object_dict) == 1:
                        obj_locations[obj_name + "_" + str(1)] = object_dict
                        obj_locations[obj_name + "_" + str(2)] = [current_loc]
                      else:
                        duplicate = object_dict.pop() 
                        prev = object_dict.pop()
                        #determine which of the two duplicates is closer to the location of that obj in the previous frame 
                        closer, further = get_closer_further(current_loc, duplicate, prev)
                        if len(object_dict) == 0:
                          obj_locations[obj_name + "_" + str(1)] = [prev, closer]
                        else:
                          obj_locations[obj_name + "_" + str(1)] = object_dict.append(prev).append(closer) 
                        obj_locations[obj_name + "_" + str(duplicate_num)] = [further]
                    else: #if this is NOT the first duplicate for this object in this timestep (case 1b2)
                      current_loc = [(x+w)//2, (y+h)//2]
                      #pop out all other duplicates and prev from 1
                      duplicates = []
                      for m in range(1, duplicate_num):
                        duplicates.append(obj_locations.pop(obj_name + "_" + str(m)).pop())
                      prev = obj_locations.pop(obj_name + "_" + str(1)).pop()
                      array_of_points = duplicates.append(current_loc)
                      closest, others_array, closest_index = get_closest(array_of_points, prev)
                      obj_locations[obj_name + "_" + str(1)].append(prev).append(closest) 
                      for n in range(2, duplicate_num + 1):
                        obj_locations[obj_name + "_" + str(n)] = [others_array[n-2]] 
                else:
                  current_frame_obj_count[obj_name] = 1
                  duplicate_check_list = []
                  for k in range(len(list(obj_locations.keys()))):
                    duplicate_check_list.append(list(obj_locations.keys())[k][0:-2])
                  if obj_name in obj_locations: #case 2
                    #append to existing key array in dictionary
                    obj_locations[obj_name].append([(x+w)//2, (y+h)//2])
                  elif obj_name in duplicate_check_list: #still case 2, just with an obj that had a duplicate in a prev timestep
                    #access (don't pop) the values for any keys where key[0:-2] matches obj_name and see which is closest to the location of obj_name, append obj_name to that one
                    current_loc = [(x+w)//2, (y+h)//2]
                    array_of_points = []
                    for p in range(len(list(obj_locations.keys()))):
                      if obj_name in list(obj_locations.keys())[p]:
                        array_of_points.append(obj_locations[list(obj_locations.keys())[p]][-1])
                    closest, others_array, closest_index = get_closest(array_of_points, current_loc)
                    obj_locations[obj_name + "_" + str(closest_index + 1)].append(current_loc)
                  else: #case 3
                    #add new key to dictionary and append location to its array
                    obj_locations[obj_name] = [[(x+w)//2, (y+h)//2]]
              elif box_type == 'side_box':
                  image = draw_side_box(image, x,y, w,h, new_color, thickness)
              else:
                  assert False

              text_color = [val for val in list(new_color)]  # saturated color version (255,255,255, 255)
              text_color[-1] /= 5 # reduce alpha
  #             Draw background rectangle (text)
              if box_type == 'box':
                  image = cv2.rectangle(image, (x, y-15), (x + (w - x), y), new_color, -1)

              # Add text
              text_color[-1] *= 5
              if box_type == 'center_dot' or 'center_dot_lined' or 'center_dot_lined_updated':
                  new_text = detic_dict[index]["values"][i]["label"]
                  image = cv2.putText(image, new_text, (int((x+w)/2-len(new_text)*2.5),(y+h)//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
              elif box_type == 'box':
                  image = cv2.putText(image, detic_dict[index]["values"][i]["label"], (x + 2,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255, 255), 1)
              elif box_type == 'side_box':
                  image = cv2.putText(image, new_text, (x-5-len(new_text)*2.5,(y+h)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
              else:
                  image = cv2.putText(image, detic_dict[index]["values"][i]["label"], (x + 2,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
            
            elif color_scheme == 'object':
              if obj_name not in color_object_dict:
                color_object_dict[obj_name] = colors_list.pop() 
              new_color = rgb_to_bgr(hex_to_rgb(color_object_dict[obj_name])) + [255] 
              image = draw_center_point_updated(image, x,y, w,h, new_color, thickness)
              if obj_name in current_frame_obj_count: #case 1
                current_frame_obj_count[obj_name] = current_frame_obj_count[obj_name] + 1
                duplicate_num = current_frame_obj_count[obj_name]
                if f == 0:
                  if duplicate_num == 2:
                    obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name)
                  obj_locations[obj_name + "_" + str(duplicate_num)] = [[(x+w)//2, (y+h)//2]]
                else:
                  if duplicate_num == 2: #if this is the first duplicate for this object in this timestep (case 1b1)
                      current_loc = [(x+w)//2, (y+h)//2]
                      duplicate_check_list = []
                      for k in range(len(list(obj_locations.keys()))):
                        duplicate_check_list.append(list(obj_locations.keys())[k][0:-2])
                      if obj_name in obj_locations: #case 2
                        if len(obj_locations[obj_name]) == 1:
                          obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name)
                          obj_locations[obj_name + "_" + str(2)] = [current_loc]
                        else:
                          duplicate = obj_locations[obj_name].pop() 
                          prev = obj_locations[obj_name][-1]
                          #determine which of the two duplicates is closer to the location of that obj in the previous frame 
                          closer, further = get_closer_further(current_loc, duplicate, prev)
                          obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name)
                          obj_locations[obj_name + "_" + str(1)].append(closer)
                          obj_locations[obj_name + "_" + str(duplicate_num)] = [further]
                      elif obj_name in duplicate_check_list: #was duplicate in prev timestep
                        if len(obj_locations[obj_name + "_" + str(1)]) == 1:
                          obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name)
                          obj_locations[obj_name + "_" + str(2)] = [current_loc]
                        else:
                          duplicate = obj_locations[obj_name + "_" + str(1)].pop() 
                          prev = obj_locations[obj_name + "_" + str(1)][-1]
                          #determine which of the two duplicates is closer to the location of that obj in the previous frame 
                          closer, further = get_closer_further(current_loc, duplicate, prev)
                          obj_locations[obj_name + "_" + str(1)] = obj_locations.pop(obj_name + "_" + str(1))
                          obj_locations[obj_name + "_" + str(1)].append(closer)
                          obj_locations[obj_name + "_" + str(duplicate_num)] = [further]
                  elif duplicate_num == 3: #if this is NOT the first duplicate for this object in this timestep (case 1b2)
                    current_loc = [(x+w)//2, (y+h)//2]
                    #pop out all other duplicates and prev from 1
                    duplicate_1 = obj_locations[obj_name + "_" + str(1)].pop()
                    duplicate_2 = obj_locations[obj_name + "_" + str(2)].pop()
                    prev = obj_locations[obj_name + "_" + str(1)][-1]
                    array_of_points = [duplicate_1, duplicate_2, current_loc]
                    closest, others_array, closest_index = get_closest(array_of_points, prev)
                    obj_locations[obj_name + "_" + str(1)].append(closest) 
                    obj_locations[obj_name + "_" + str(2)] = [others_array[0]] 
                    obj_locations[obj_name + "_" + str(3)] = [others_array[1]] 
              else:
                current_frame_obj_count[obj_name] = 1
                duplicate_check_list = []
                for k in range(len(list(obj_locations.keys()))):
                  duplicate_check_list.append(list(obj_locations.keys())[k][0:-2])
                if obj_name in obj_locations: #case 2
                  #append to existing key array in dictionary
                  obj_locations[obj_name].append([(x+w)//2, (y+h)//2])
                elif obj_name in duplicate_check_list: #still case 2, just with an obj that had a duplicate in a prev timestep
                  #access (don't pop) the values for any keys where key[0:-2] matches obj_name and see which is closest to the location of obj_name, append obj_name to that one
                  current_loc = [(x+w)//2, (y+h)//2]
                  array_of_points = []
                  for p in range(len(list(obj_locations.keys()))):
                    if obj_name in list(obj_locations.keys())[p]:
                      array_of_points.append(obj_locations[list(obj_locations.keys())[p]][-1])
                  closest, others_array, closest_index = get_closest(array_of_points, current_loc)
                  obj_locations[obj_name + "_" + str(closest_index + 1)].append(current_loc)
                else: #case 3
                  #add new key to dictionary and append location to its array
                  obj_locations[obj_name] = [[(x+w)//2, (y+h)//2]]
              
              text_color = [val for val in list(new_color)]  # saturated color version (255,255,255, 255)
              text_color[-1] /= 5 # reduce alpha
  #             Draw background rectangle (text)

              # Add text
              text_color[-1] *= 5
              if box_type == 'center_dot' or 'center_dot_lined' or 'center_dot_lined_updated':
                  new_text = detic_dict[index]["values"][i]["label"]
                  image = cv2.putText(image, new_text, (int((x+w)/2-len(new_text)*2.5),(y+h)//2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
              elif box_type == 'box':
                  image = cv2.putText(image, detic_dict[index]["values"][i]["label"], (x + 2,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255, 255), 1)
              elif box_type == 'side_box':
                  image = cv2.putText(image, new_text, (x-5-len(new_text)*2.5,(y+h)//2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
              else:
                  image = cv2.putText(image, detic_dict[index]["values"][i]["label"], (x + 2,y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)

          
          #current_frame_obj_counts_all.append(current_frame_obj_count)

    #now we draw all lines between dots 
    if len(index_list) > 1:
      for i in range(len(list(obj_locations.keys()))): #for each distinct object in the video
        if len(obj_locations[list(obj_locations.keys())[i]]) == 1:
          continue
        else:
          for j in range(1, len(obj_locations[list(obj_locations.keys())[i]])): #for each location of object with label list(obj_locations.keys())[i] 
            obj_name = list(obj_locations.keys())[i]
            if list(obj_locations.keys())[i][-2] == '_':
              obj_name = obj_name[0:-2]
            arrow_color = rgb_to_bgr(hex_to_rgb(color_object_dict[obj_name])) + [255]
            image = draw_arrow(image, obj_locations[list(obj_locations.keys())[i]][j-1][0], obj_locations[list(obj_locations.keys())[i]][j-1][1], obj_locations[list(obj_locations.keys())[i]][j][0], obj_locations[list(obj_locations.keys())[i]][j][1], arrow_color, 2)


    return image