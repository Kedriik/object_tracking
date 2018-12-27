# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import win32api
import time
import math
#import zipfile

#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image
from selectscreen import select_screen
print("Select screen")
pos  = select_screen()

from grabscreen import grab_screen
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

import label_map_util
import visualization_utils as vis_util


# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT =  'frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# =============================================================================
# # ## Download Model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())
# =============================================================================


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap( 'mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def screen_distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))

def compute_centroid(bbox):
      mid_x = (bbox[3] + bbox[1]) / 2
      mid_y = (bbox[2] + bbox[0]) / 2
      point = []
      point.append(mid_x)
      point.append(mid_y)
      return point
    
capturing_coordinates = (pos[0],pos[1],pos[2],pos[3]) #(100,14,800,600)
capturing_size = (pos[2]-pos[0],pos[3]-pos[1])
capturing_resize = (pos[2]-pos[0],pos[3]-pos[1])


running_trackers = []
max_trackers = 0

def bboxToPixels(bbox):
    xmin = int(bbox[1]*float(capturing_size[0]))
    ymin = int(bbox[0]*float(capturing_size[1]))
    xmax = int(bbox[3]*float(capturing_size[0]))
    ymax = int(bbox[2]*float(capturing_size[1]))
    return (xmin,ymin,xmax-xmin,ymax-ymin)

def isPointInsideBbox(point,bbox):
    xmin = bbox[1]
    ymin = bbox[0]
    xmax = bbox[3]
    ymax = bbox[2]
    if point[0] > xmin and point[0] < xmax and point[1] > ymin and point[1] < ymax:
        return True
    else:
        return False

def showTrackedItems(frame,bbox):
    crop_img = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
    cv2.imshow("cropped", crop_img)
saved_images_counter = 0
def saveImage(image):
    global images_params
    global saved_images_counter
    cv2.imwrite('D:\Temporary\TestDir\test{}.jpg'.format(saved_images_counter)
    , image)
    saved_images_counter = saved_images_counter + 1

tracked_items = []
tracked_items_offscreen_frames = []
tracked_ids = []
tracker_ID = 0
           
def add_tracker(frame,bbox):
    global tracker_ID
    bbox = bboxToPixels(bbox)
    #TLD the best so far
    #tracker = cv2.TrackerTLD_create()
    tracker = cv2.TrackerTLD_create()
    tracker.init(frame, bbox)
    print("tracker added")
    running_trackers.append(tracker)
    tracked_ids.append(tracker_ID)
    tracker_ID =  tracker_ID + 1
    print(bbox)
    showTrackedItems(frame,bbox)

def remove_tracker(index):
    del running_trackers[index]
    del tracked_items[index]
    del tracked_items_offscreen_frames[index]
    del tracked_ids[index]
    
#def saveTrackedItemAsPNG(frame, bbox):
    


# Size, in inches, of the output images.

IMAGE_SIZE = (12, 8)
mouseX = 0
mouseY = 0
detected_items = []  
def track_selected(event,x,y,flags,param):
    global detected_items
    global image_np
    if event == cv2.EVENT_LBUTTONDBLCLK:
        click_point = []
        click_point.append(x/capturing_resize[0]);
        click_point.append(y/capturing_resize[1]);
        for i in range(len(detected_items)):
            if isPointInsideBbox(click_point, detected_items[i]) == True:
                print("Inside")
                add_tracker(image_np,detected_items[i])
                
cv2.namedWindow('window')
cv2.setMouseCallback('window',track_selected)                

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      state_right = win32api.GetKeyState(0x02)
      if state_right == 0 or state_right == 1:
          screen = cv2.resize(grab_screen(region=(capturing_coordinates)), (capturing_resize))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      ##### boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax)
                  
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=1)
      #center points of detected centroids
      
      detected_items.clear();
      for i,b in enumerate(boxes[0]):
          if scores[0][i] > 0.33:# and classes[0][i] == 3:
              detected_items.append(b.tolist())
      for i in range(len(running_trackers)):
              # Update tracker
          ok, bbox = running_trackers[i].update(image_np)
              # Draw bounding box
          if ok:
              # Tracking success
              p1 = (int(bbox[0]), int(bbox[1]))
              p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
              cv2.rectangle(image_np, p1, p2, (255,255,255), 3, 1)
#              cv2.putText(image_np, 'ID={}'.format(tracked_ids[i]), 
#                          (int(capturing_size[0]*compute_centroid(tracked_items[i])[0]),int(capturing_size[1]*compute_centroid(tracked_items[i])[1])),
#                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,0,0),2)
          else :
              # Tracking failure
              cv2.putText(image_np, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
      #state_right = win32api.GetKeyState(0x02)
      #print(state_right)
      cv2.imshow('window',image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break