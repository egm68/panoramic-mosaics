[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a_U4QmuLPnrFZvHjSu_vRImLHaS63rKC?usp=sharing) [![Paper](https://img.shields.io/badge/read%20the-paper-blueviolet)](https://github.com/egm68/panoramic-mosaics/blob/main/panoMosaics_report.pdf) 

# panoMosaics
panoMosaics is a Python library that allows users to create enhanced object detection visualizations that capture multiple timesteps using panoramic mosaics. 

### Install
panoMosaics can be installed using the command

```shell
pip install git+https://github.com/egm68/panoramic-mosaics
```
### Usage 
You can easily visualize the output of an object detection model over several frames, as shown below

```python
import panoMosaics

#load video
main_frame_arr = panoMosaics.video_to_frame_arr('2023.03.29-17.39.21-main.avi')
     
#load object detection model output
with open('2023.03.29-17.39.21-detic:image.json', 'r') as j:
     detic_dict = json.loads(j.read())
     
#gets the # of seconds from start for each frame
frame_sfs_arr = panoMosaics.get_sfs_arr(main_frame_arr)

#match frames with timestamps 
timestamps_sfs_arr = panoMosaics.get_timestamp_arr(detic_dict)
frames_timestamps_arr = panoMosaics.get_frame_timestamps_arr(main_frame_arr, detic_dict, frame_sfs_arr, timestamps_sfs_arr)

#stitch panorama using specified frames
src_list = [main_frame_arr[160]]
src_index_list = [160]
dst = main_frame_arr[196]
kp_dst, des_dst = panoMosaics.get_keypoints_descriptors(dst)
transf_list = []
for i in range(len(src_list)):
  kp_src, des_src = panoMosaics.get_keypoints_descriptors(src_list[i])
  matches = panoMosaics.feature_matching(des_src, des_dst)
  transf = panoMosaics.get_homography_matrix(src_list[i], dst, kp_src, kp_dst, matches, 4)
  transf_list.append(transf)
dst_pad, warped_src_arr, new_transf_list, anchorX, anchorY = panoMosaics.warp_n_with_padding(dst, src_list, transf_list, main_frame_arr)
im_arr = panoMosaics.get_rgba_im_arr(dst_pad, warped_src_arr)
comp_arr = panoMosaics.alpha_composite_n_images_parallel(im_arr)

#add object detection visualization
transf_index_dict = {}
for i in range(len(src_index_list)):
  transf_index_dict[src_index_list[i]] = new_transf_list[i]
  
pano_with_bounding_boxes = panoMosaics.draw_all_bounding_boxes_for_given_indices([160, 196], frames_timestamps_arr, 
                                                                     detic_dict, comp_arr, transf_index_dict, 196,
                                                                     anchorX, anchorY, ["#e41a1c","#377eb8","#d920f5","#ff7f00","#ffff33", 
                                                                     "#00ff00d9", "#17becf", "#2323d9", '#0e9620'], "object", 2, 
                                                                     "arrow", [])
```
![a panoramic mosaic output by above code block](https://github.com/egm68/panoramic-mosaics/blob/main/results/final_pano_frames/pano-with-arrows-colorobject.png?raw=true)

### Try it out yourself!
Try our demo on Colab [here](https://colab.research.google.com/drive/1a_U4QmuLPnrFZvHjSu_vRImLHaS63rKC?usp=sharing).

### Want to learn more?
Read our project report [here](https://github.com/egm68/panoramic-mosaics/blob/main/panoMosaics_report.pdf). 
