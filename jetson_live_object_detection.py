import tensorflow as tf
import numpy as np
import argparse
import cv2
import tensorflow.contrib.tensorrt as trt
import time
import sys
import signal
from src.object_detector import ObjectDetection
import os
import datetime
from submarine_msgs_srvs.msg import Detections


""" Jetson Live Object Detector """
class JetsonLiveObjectDetection():
    def __init__(self, model, camera=None, debug=False, thresh=0.4, last_network_callback_time=0.0):
        self.debug = debug
        if test_video_picture is not None:
            self.camera = cv2.VideoCapture(test_video_picture)    
        elif self.debug:    
            self.camera = cv2.VideoCapture(camera)
        self.model = model
        self.detector = ObjectDetection(self.model, label_map=args.label)
        self.thresh = float(thresh)
        self.last_network_callback_time = last_network_callback_time

    def signal_handler(self, sig, frame):
        cv2.destroyAllWindows()
        if args.debug or test_video_picture is not None:
            self.camera.release()
        self.detector.__del__()
        sys.exit(0)

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
                x = int(bbox[1] * cols)
                y = int(bbox[0] * rows)
                right = int(bbox[3] * cols)
                bottom = int(bbox[2] * rows)
                thickness = int(4 * score)
                cv2.rectangle(img, (x, y), (right, bottom), (125,255, 21), thickness=thickness)
                cv2.putText(img, self.detector.labels[str(classId)] + ': ' + str(round(score,3)), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        return img, detections

    def static_video(self):
        if test_video_picture is not None:
            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            names = test_video_picture.split('.')
            if args.test_video is not None and not args.no_save_images:
                out = cv2.VideoWriter(names[0] + '_output.' + names[1], fourcc, 30.0, (640,480))
            while(self.camera.isOpened()):
                ret, img = self.camera.read()
                if ret:
                    scores, boxes, classes, num_detections = self.detector.detect(img)
                    img, new_detections = self._visualizeDetections(img, scores, boxes, classes, num_detections)
                    if args.test_video is not None:
                        img = cv2.resize(img, (640,480))
                    if (not args.no_video):
                        cv2.imshow(names[0], img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    if args.test_video is not None and not args.no_save_images:
                        out.write(img)
                    elif args.test_picture is not None:
                        cv2.imwrite(names[0] + '_output.' + names[1], img)
                    print ("Found objects: " + str(' '.join(new_detections)) + ".")
                else:
                    break
            if args.test_video is not None and not args.no_save_images:
                out.release()
            self.camera.release()
            self.detector.__del__()
            if (not args.no_save_images):
                print("Output File written to " + names[0] + "_output." + names[1])
            exit()

    def start(self):
        img_counter = 0
        frame_counter = 0

        if (not args.no_save_images):
            script_directory = os.path.dirname(os.path.realpath(__file__)) + '/'
            save_dir = script_directory + 'saved_video/{}_{}/'.format(self.model,datetime.datetime.now())
            os.mkdir(save_dir)

        print ("Starting Live object detection, may take a few minutes to initialize...")
        self.detector.initializeSession()
        
        # Run the code with static video/picture testing:
        if test_video_picture is not None:
           self.static_video() 

        # Run the code in DEBUG mode. That is, with a local camera.
        elif self.debug:  
            # Health Checks:
            if not self.camera.isOpened():
                print ("Camera has failed to open")
                exit(-1)

            fps = self.camera.get(cv2.CAP_PROP_FPS)

            # Main Programming Loop
            while True:
                curr_time = time.time()

                ret, img = self.camera.read()
                if (frame_counter >= fps / args.rate):
                    frame_counter = 0
                    scores, boxes, classes, num_detections = self.detector.detect(img)
                    new_detections = None
                    img, new_detections = self._visualizeDetections(img, scores, boxes, classes, num_detections)
            
                    print ("Found objects: " + str(' '.join(new_detections)) + ".")
                    if (not args.no_video):
                        cv2.imshow('Jetson Live Detection', img)
                    if ((img_counter % 3) == 0 and not args.no_save_images):
                        img_name = "{}opencv_frame_{}.jpg".format(save_dir, int(curr_time))
                        cv2.imwrite(img_name, img)
                        img_counter = 0
                
                    img_counter += 1

                    # Publish ros-bridged images
                    if not args.no_ros:
                        img_msg = bridge.cv2_to_imgmsg(img)
                        img_pub.publish(img_msg)

                        detections_msg = Detections()
                        detections_msg.scores = scores
                        detections_msg.boxes = boxes.flatten()
                        detections_msg.classes = classes
                        detections_msg.detected = [num_detections]
                        detections_pub.publish(detections_msg)

                    print ("Network running at: " + str(1.0/(time.time() - curr_time)) + " Hz.")

                frame_counter += 1
            
                if cv2.waitKey(1) == ord('q'):
                    break

            cv2.destroyAllWindows()
            self.camera.release()
            self.detector.__del__()
            print ("Exiting...")
            return

        # Run the code as a ROS node, pulls images on a topic, published them out on antoher
        else:
            rospy.Subscriber('raw_imgs', Image, self.run_network_node)
            rospy.spin()

    def run_network_node(self, msg):
        if (time.time() - self.last_network_callback_time) <= args.rate:
            return
        bridge = cv_bridge.CvBridge()
        img = bridge.imgmsg_to_cv2(msg)

        scores, boxes, classes, num_detections = self.detector.detect(img)
        new_detections = None
        img, new_detections = self._visualizeDetections(img, scores, boxes, classes, num_detections)
        print ("Found objects: " + str(' '.join(new_detections)) + ".")

        img_msg = bridge.cv2_to_imgmsg(img)
        img_pub.publish(img_msg)

        detections_msg = Detections()
        detections_msg.scores = scores
        detections_msg.boxes = boxes.flatten()
        detections_msg.classes = classes
        detections_msg.detected = [num_detections]
        detections_pub.publish(detections_msg)

        self.last_network_callback_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This script runs inference on a trained object detection network")
    parser.add_argument('-m', '--model', default="ssd_mobilenet_v1_coco", help="set name of neural network model to use")
    parser.add_argument('-v', '--verbosity', action='store_true', help="set logging verbosity (doesn't work)")
    parser.add_argument('-d', '--debug', action='store_true', help='Runs the network using a local camera, not from ROS, but will still publish to ROS topics.')
    parser.add_argument('-c', '--camera', default='/dev/video0', help='/path/to/video, defaults to /dev/video0')
    parser.add_argument('-r', '--rate', type=int, default=-1, help='Specify the rate to run the neural network at, i.e. number of images to look at per second. Defaults to fastests possible.')
    parser.add_argument('-l', '--label', default='label_map.pbtxt', help='Override the name of the label map in your model directory. Defaults to label_map.pbtxt')
    parser.add_argument('--test-video', help='/path/to/test/video This is used if you want to test your network on a static video. It will append \'_output\' to your file before saving it.')
    parser.add_argument('--test-picture', help='/path/to/test/picture This is used if you want to test your network on a static picture. It will append \'_output\' to your file before saving it.')
    parser.add_argument('--thresh', default=0.4, help='Override the default detection threshold. Default = 0.4')
    parser.add_argument('--no-video', action='store_true', help='Will not display live video feed, will still display in terminal.')
    parser.add_argument('--no-save-images', action='store_true', help='Will not record any video/pictures from the sub')
    parser.add_argument('--no-ros', action='store_true', help='Will not subscribe or publish to any ros topics')

    args = parser.parse_args()

    if not args.no_ros:
        import cv_bridge
        import rospy
        from sensor_msgs.msg import Image
        rospy.init_node('Network_Vision')
        img_pub = rospy.Publisher('network_imgs', Image, queue_size=1)
        bridge = cv_bridge.CvBridge()
        detections_pub = rospy.Publisher('network_output', Detections, queue_size=1)

    test_video_picture = None
    if args.test_video is not None and args.test_picture is not None:
        print("Please don't use --test_video and --test_picture at the same time.")
        exit()
    elif args.test_video is not None:
        test_video_picture = args.test_video
    elif args.test_picture is not None:
        test_video_picture = args.test_picture

    live_detection = JetsonLiveObjectDetection(model=args.model, camera=args.camera, debug=args.debug, thresh=args.thresh, last_network_callback_time=time.time())

    # captureing Ctrl+C
    signal.signal(signal.SIGINT, live_detection.signal_handler)

    live_detection.start()
    

