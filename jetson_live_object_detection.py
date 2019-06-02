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
    def __init__(self, model, camera, debug=False, thresh=0.4, fps = 10.0):
        self.debug = debug
        if test_video_picture is not None:
            self.camera = cv2.VideoCapture(test_video_picture)    
        else:    
            self.camera = cv2.VideoCapture(camera)
        self.model = model
        self.rate = float(1.0 / fps)
        self.detector = ObjectDetection(self.model, label_map=args.label)
        self.thresh = float(thresh)

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
        if (not args.noVideo and test_video_picture is None):
            cv2.imshow('Jetson Live Detection', img)

        return img

    def start(self):
        print ("Starting Live object detection, may take a few minutes to initialize...")
        self.detector.initializeSession()

        # For static video/picture testing:
        if test_video_picture is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            names = test_video_picture.split('.')
            if args.testVideo is not None:
                out = cv2.VideoWriter(names[0] + '_output.' + names[1], fourcc, 20.0, (640,480))
            while(self.camera.isOpened()):
                ret, img = self.camera.read()
                if ret:
                    scores, boxes, classes, num_detections = self.detector.detect(img)
                    img = self._visualizeDetections(img, scores, boxes, classes, num_detections)
                    if args.testVideo is not None:
                        out.write(img)
                        cv2.imshow(names[0],img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    elif args.testPicture is not None:
                        cv2.imwrite(names[0] + '_output.' + names[1], img)
                else:
                    break
            if args.testVideo is not None:
                out.release()
            self.camera.release()
            self.detector.__del__()
            print("Output File written to " + names[0] + "_output." + names[1])
            exit()

        # Health Checks:
        if not self.camera.isOpened():
            print ("Camera has failed to open")
            exit(-1)
        elif not self.debug:
            #TODO: Add ROS setup stuff here
            print("TODO")
    
        # Main Programming Loop
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
    parser.add_argument('-m', '--model', default="ssd_mobilenet_v1_coco", help="set name of neural network model to use")
    parser.add_argument('-v', '--verbosity', action='store_true', help="set logging verbosity (doesn't work)")
    parser.add_argument('-d', '--debug', action='store_true', help='Runs only the network without ROS. (doesn\'t work)')
    parser.add_argument('-c', '--camera', default='/dev/video0', help='/path/to/video, defaults to /dev/video0')
    parser.add_argument('-l', '--label', default='label_map.pbtxt', help='Override the name of the label map in your model directory. Defaults to label_map.pbtxt')
    parser.add_argument('--testVideo', help='/path/to/test/video This is used if you want to test your network on a static video. It will append \'_output\' to your file before saving it.')
    parser.add_argument('--testPicture', help='/path/to/test/picture This is used if you want to test your network on a static picture. It will append \'_output\' to your file before saving it.')
    parser.add_argument('--thresh', default=0.4, help='Override the default detection threshold. Default = 0.4')
    parser.add_argument('--noVideo', action='store_true', help='Will not display live video feed, will still display in terminal.')
    
    args = parser.parse_args()

    test_video_picture = None
    if args.testVideo is not None and args.testPicture is not None:
        print("Please don't use --testVideo and --testPicture at the same time.")
        exit()
    elif args.testVideo is not None:
        test_video_picture = args.testVideo
    elif args.testPicture is not None:
        test_video_picture = args.testPicture

    live_detection = JetsonLiveObjectDetection(model=args.model, camera=args.camera, debug=args.debug, thresh=args.thresh, fps=10.0)
    live_detection.start()
    

