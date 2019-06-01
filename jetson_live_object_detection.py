import tensorflow as tf
import numpy as np
import argparse
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from src.object_detector import ObjectDetection

""" Jetson Live Object Detector """
class JetsonLiveObjectDetection():
    def __init__(self, model, camera, debug=False, thresh=0.4, fps = 10.):
        self.debug = debug
        self.camera = cv2.VideoCapture(camera)
        self.model = model
        self.rate = float(1. / fps)
        self.detector = ObjectDetection('./data/' + self.model + '/' + self.model + '_trt_graph.pb')
        self.thresh = thresh

    def _visualizeDetections(self, img, scores, boxes, classes, num_detections):
        cols = img.shape[1]
        rows = img.shape[0]
        detections = []

        for i in range(num_detections):
            bbox = [float(p) for p in boxes[i]]
            score = float(scores[i])
            classId = int(classes[i])
            if score > self.thresh:
                detections.append(self.detector.labels[str(classId)])
                if (not args.noVideo):
                    x = int(bbox[1] * cols)
                    y = int(bbox[0] * rows)
                    right = int(bbox[3] * cols)
                    bottom = int(bbox[2] * rows)
                    thickness = int(4 * score)
                    cv2.rectangle(img, (x, y), (right, bottom), (125,255, 21), thickness=thickness)
                    cv2.putText(img, self.detector.labels[str(classId)] + ': ' + str(round(score,3)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        print ("Found objects: " + str(' '.join(detections)) + ".")
        if (not args.noVideo):
            cv2.imshow('Jetson Live Detection', img)

    def start(self):
        print ("Starting Live object detection, may take a few minutes to initialize...")
        self.detector.initializeSession()

        if not self.camera.isOpened():
            print ("Camera has failed to open")
            exit(-1)
        elif not self.debug:
            #TODO: Add ROS setup stuff here
            print("TODO")
    
        while True:
            curr_time = time.time()

            ret, img = self.camera.read()
            scores, boxes, classes, num_detections = self.detector.detect(img)

            self._visualizeDetections(img, scores, boxes, classes, num_detections)
            
            if not self.debug:
                #TODO: Add runtime ros publishers
                print("TODO")

            print ("Running at: " + str(1.0/(time.time() - curr_time)) + " Hz.")

            if cv2.waitKey(1) == ord('q'):
                break

            # throttle to rate
            capture_duration = time.time() - curr_time
            sleep_time = self.rate - capture_duration
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        cv2.destroyAllWindows()
        self.camera.release()
        self.detector.__del__()
        print ("Exiting...")
        return



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script runs inference on a trained object detection network")
    parser.add_argument('-t', '--training_set', help="specify set of targets to use (doesn't work)")
    parser.add_argument('-n', '--network', default="ssd_mobilenet_v1_coco", help="set name of neural network graph to use")
    parser.add_argument('-v', '--verbosity', action='store_true', help="set logging verbosity (doesn't work)")
    parser.add_argument('-d', '--debug', action='store_true', help='Runs only the network without ROS. (doesn\'t work)')
    parser.add_argument('-c', '--camera', default='/dev/video0', help='/path/to/video, defaults to /dev/video0')
    parser.add_argument('--thresh', default=0.4, help='Override the default detection threshold. Default = 0.4')
    parser.add_argument('--noVideo', action='store_true', help='Will not display live video feed, will still display in terminal.')
    
    args = parser.parse_args()

    live_detection = JetsonLiveObjectDetection(model=args.network, camera=args.camera, debug=args.debug, thresh=args.thresh, fps=10.0)
    live_detection.start()
    

