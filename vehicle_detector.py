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
gp1 = [10,0]
gp2 = [0,10]
def screen_distance(p1, p2):
    #gp1=p1
    #gp2=p2
    return math.sqrt(math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))

running_trackers = []
max_trackers = 10
def add_tracker(frame,bbox):
    if len(running_trackers) < max_trackers:
        tracker = cv2.TrackerGOTURN_create()
        tracker.init(frame, bbox)
        running_trackers.append(tracker)
    

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

capturing_coordinates = (pos[0],pos[1],pos[2],pos[3]) #(100,14,800,600)
capturing_size = (capturing_coordinates[2],capturing_coordinates[3])
capturing_resize = (capturing_coordinates[2],capturing_coordinates[3])

tracked_centroids = []
ID = 0

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
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
      currently_detected_centroids = []
      
      for i,b in enumerate(boxes[0]):
          if scores[0][i] > 0.5:
              mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
              mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
              point = []
              point.append(mid_x)
              point.append(mid_y)
              currently_detected_centroids.append(point)
      
      for i in range(len(tracked_centroids)):
          min_dist = -1
          for j in range(len(currently_detected_centroids)):
              dist = screen_distance(tracked_centroids[i] , currently_detected_centroids[j])
              if min_dist == -1 or dist < min_dist:
                  min_dist = dist
                  min_centroid = currently_detected_centroids[j]
          tracked_centroids[i] = min_centroid
          currently_detected_centroids.remove(min_centroid)
          
      left_bboxes = []
      for i,b in enumerate(boxes[0]):
          if scores[0][i] > 0.5:
              mid_x = (boxes[0][i][3] + boxes[0][i][1]) / 2
              mid_y = (boxes[0][i][2] + boxes[0][i][0]) / 2
              point = []
              point.append(mid_x)
              point.append(mid_y)
              for j in range(len(currently_detected_centroids)):
                  if screen_distance(point, currently_detected_centroids[j]):
                      left_bboxes.append(b)
      
      for i in range(len(left_bboxes)):
          add_tracker(image_np,left_bboxes[i])
          
      for i in range(len(running_trackers)):
              # Update tracker
          ok, bbox = running_trackers[i].update(image_np)
              # Draw bounding box
          if ok:
              # Tracking success
              p1 = (int(bbox[0]), int(bbox[1]))
              p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
              cv2.rectangle(image_np, p1, p2, (255,0,0), 2, 1)
          else :
              # Tracking failure
              cv2.putText(image_np, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

      cv2.imshow('window',image_np)
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break