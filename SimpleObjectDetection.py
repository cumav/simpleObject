import cv2 as cv
import json


class SimpleObjectDetection:

    def __init__(self, inference_graph='./model/frozen_inference_graph.pb',
                 pbtxt='./model/graph.pbtxt'):
        # All detections get stored here
        self.detections = []
        # opencv types
        self.counter = 0
        self.fontColor = (23, 230, 210)
        self.fontScale = 0.5
        self.lineType = 2
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.frame_cnt = 0

        # Open labelmap
        with open("name_labels.json", "r") as labels:
            self.label_data = json.load(labels)

        # Open the NN
        self.cvNet = cv.dnn.readNetFromTensorflow(inference_graph, pbtxt)

    def detect(self, frame):
        '''
        Detect features from a frame.
        :param frame: Well its a frame
        :return:
        '''
        self.cvNet.setInput(
            cv.dnn.blobFromImage(frame, size=(300, 300), swapRB=True,
                                 crop=False))
        cvOut = self.cvNet.forward()
        self.detections = cvOut[0, 0, :, :]

    def frame_from_image(self, image_path):
        '''
        Load an image.
        :param image_path: path to the image.
        '''
        return cv.imread(image_path)

    def draw_bounding_boxes(self, frame, min_detection_score=0.3, save_path="",
                            plot=False, used_ids=[]):
        '''
        Function to draw bounding boxes in a frame.
        :param frame: The frame to draw the bounding boxes in
        :param min_detection_score: the minimum score needed to trigger bounding box generation.
        :param save_path: path to where the image should be saved. If empty it wont get saved.
        :param plot: (Bool) If true will plot the given frame after writing bounding boxes.
        :param used_ids: Allowed ids of which bounding boxes should be drawn. If empty all ids will get used.
        '''
        rows = frame.shape[0]
        cols = frame.shape[1]
        for detection in self.detections:
            score = float(detection[2])

            label_id = int(detection[1])
            if score > min_detection_score and (
                    label_id in used_ids or used_ids == []):
                name = self.label_data[str(label_id)]

                left = detection[3] * cols
                top = detection[4] * rows
                right = detection[5] * cols
                bottom = detection[6] * rows
                cv.rectangle(frame, (int(left), int(top)),
                             (int(right), int(bottom)), self.fontColor,
                             thickness=self.lineType)

                bottomLeftCornerOfText = (int(left), int(bottom))
                cv.putText(frame, name,
                           bottomLeftCornerOfText,
                           self.font,
                           self.fontScale,
                           self.fontColor,
                           self.lineType)

            if save_path != "":
                cv.imwrite(save_path, frame)
        if plot and self.frame_cnt % 30 == 0:
            cv.imshow('img', frame)
            if cv.waitKey(25) & 0xFF == ord('q'):
                pass

    def image_boxes(self, path, save_path="", plot=False):
        frame = self.frame_from_image(path)
        self.detect(frame)
        self.draw_bounding_boxes(frame, save_path=save_path, plot=plot)

    def video_boxes(self, source_path, plot=True):
        '''
        :param source_path: Either String to use video file or an int for cam use. Checkout OpenCV documentation for
        more information.
        :param plot: Bool stating if output should be shown on screen.
        '''
        cap = cv.VideoCapture(source_path)
        while cap.isOpened():
            ret, frame = cap.read()
            self.detect(frame)
            self.draw_bounding_boxes(frame, plot=plot)

    def video_function(func):
        '''
        Helper function allows to create custom functions on each frame.
        '''

        def wrapper(self, source_path):
            cap = cv.VideoCapture(source_path)
            while cap.isOpened():
                ret, frame = cap.read()
                self.detect(frame)
                func(self, frame)

        return wrapper

    def return_detection_features(self, frame, used_ids=[]):
        '''
        Detects and returns the detection information of a frame.
        :param frame: The frame  of the last detection.
        :param used_ids: Allowed ids of which to return information. If empty all ids will get used.
        :return: A dict containing every detection and its features:
        Example:
        {
         0: {'score': 0.9607022404670715, 'label_id': 1, 'name': 'person',
             'left': 276.52418613433838, 'top': 6.009276956319809,
             'right': 344.27529573440552, 'bottom': 213.1209522485733},
         1: {'score': 0.7422491908073425, 'label_id': 1, 'name': 'person',
             'left': 432.36765265464783, 'top': 123.44587594270706,
             'right': 446.73523306846619, 'bottom': 178.85421216487885},
         2: {'score': 0.295931875705719, 'label_id': 1, 'name': 'person',
             'left': 174.88788068294525, 'top': 135.12024655938148,
             'right': 197.32286036014557, 'bottom': 163.72953727841377},
         3: {'score': 0.2481938898563385, 'label_id': 1, 'name': 'person',
             'left': 481.41738772392273, 'top': 129.66636568307877,
             'right': 490.77251553535461, 'bottom': 141.54470711946487},
         4: {'score': 0.2456628531217575, 'label_id': 1, 'name': 'person',
             'left': 253.99434566497803, 'top': 32.183747738599777,
             'right': 355.22496700286865, 'bottom': 226.74961388111115},
        }
        '''
        self.detect(frame)
        detection_features = {}
        rows = frame.shape[0]
        cols = frame.shape[1]
        for cnt, detection in enumerate(self.detections):
            detection_features[cnt] = {}
            detection_features[cnt]["score"] = float(detection[2])

            detection_features[cnt]["label_id"] = int(detection[1])
            if (detection_features[cnt][
                "label_id"] in used_ids or used_ids == []):
                detection_features[cnt]["name"] = self.label_data[
                    str(detection_features[cnt]["label_id"])]

                detection_features[cnt]["left"] = detection[3] * cols
                detection_features[cnt]["top"] = detection[4] * rows
                detection_features[cnt]["right"] = detection[5] * cols
                detection_features[cnt]["bottom"] = detection[6] * rows
        return detection_features


if __name__ == '__main__':
    # x = SimpleObjectDetection()
    # x.image_boxes("test.png")
    # x.video_boxes(0, plot=True)

    class VideoPointer(SimpleObjectDetection):

        @SimpleObjectDetection.video_function
        def testen(self, frame):
            for detection in self.detections:
                score = float(detection[2])
                if score > 0.1:
                    label_id = int(detection[1])
                    print(self.label_data[str(label_id)])
            self.draw_bounding_boxes(frame, plot=True, min_detection_score=0.2,
                                     used_ids=[1])


    a = VideoPointer()
    a.testen(source_path="test.mp4")
