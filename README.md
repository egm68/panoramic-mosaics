[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1a_U4QmuLPnrFZvHjSu_vRImLHaS63rKC?usp=sharing) [![Paper](https://img.shields.io/badge/read%20the-paper-blueviolet)](https://github.com/egm68/panoramic-mosaics/blob/main/panoMosaics_report.pdf) 

# panoMosaics
panoMosaics is a Python library that allows users to create enhanced object detection visualizations that capture multiple timesteps using panoramic mosaics. 

### Install
panoMosaics can be installed using the command

```shell
pip install git+https://github.com/egm68/panoramic-mosaics
```
### Usage 
You can easily visualize the output of an object detection model over several frames with just a few lines of code, as shown below:

```python
import panoMosaics

#load video
main_frame_arr = panoMosaics.video_to_frame_arr('2023.03.29-17.39.21-main.avi')
     
#load object detection model output
with open('2023.03.29-17.39.21-detic:image.json', 'r') as j:
     detic_dict = json.loads(j.read())

#stitch panorama using specified frames
comp_arr, frames_timestamps_arr, transf_index_dict, anchorX, anchorY = stitch_frames(main_frame_arr, detic_dict, [160, 169, 178, 187], 196)

#add object detection visualization  
pano_with_bounding_boxes = panoMosaics.draw_all_bounding_boxes_for_given_indices([160, 196], frames_timestamps_arr, 
                                                                     detic_dict, comp_arr, transf_index_dict, 196,
                                                                     anchorX, anchorY, ["#e41a1c","#377eb8","#d920f5","#ff7f00","#ffff33", 
                                                                     "#00ff00d9", "#17becf", "#2323d9", '#0e9620'], "object", 2, 
                                                                     "arrow", [])
```
Output:


![a panoramic mosaic output by above code block](https://github.com/egm68/panoramic-mosaics/blob/main/results/final_pano_frames/pano-with-arrows-colorobjects-5frames.png?raw=true)

### Try it out yourself!
Try our demo on Colab [here](https://colab.research.google.com/drive/1a_U4QmuLPnrFZvHjSu_vRImLHaS63rKC?usp=sharing).

### Want to learn more?
Read our project report [here](https://github.com/egm68/panoramic-mosaics/blob/main/panoMosaics_report.pdf). 
