import cv2
import imutils
from imutils.video import WebcamVideoStream

class pose_landmarks:
    def __init__(self, select_model='mpi'):
        self.select_model = select_model
        self.in_width = 186
        self.in_height = 186
        self.point_classes, self.point_pairs, self.n_points = self.coco() if select_model == 'coco' else self.mpi()
        if select_model == 'body25':
            self.point_pairs, self.n_points = self.body25()
        self.proto_file = {
            'coco': './pose/coco/pose_deploy_linevec.prototxt',
            'mpi': './pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt',
            'body25': './pose/body_25/post_deploy.prototxt'
        }
        self.weight_file = {
            'coco': './pose/coco/pose_iter_440000.caffemodel',
            'mpi': './pose/mpi/pose_iter_160000.caffemodel',
            'body25': './pose/body_25/pose_iter_584000.caffemodel'
        }
        self.net = cv2.dnn.readNetFromCaffe(self.proto_file[str(self.select_model)],
                                            self.weight_file[str(self.select_model)])

    def coco(self):
        point_classes = {
            "nose": 0, "neck": 1,
            "right shoulder": 2, "right elbow": 3, "right wrist": 4,
            "left shoulder": 5, "left elbow": 6, "left wrist": 7,
            "right hip": 8, "right knee": 9, "right ankle": 10,
            "left hip": 11, "left knee": 12, "left ankle": 13,
            "right eye": 14, "left eye": 15,
            "right ear": 16, "left ear": 17, }
        point_pairs = [[1, 0], [1, 2], [1, 5],
                       [2, 3], [3, 4], [5, 6],
                       [6, 7], [1, 8], [8, 9],
                       [9, 10], [1, 11], [11, 12],
                       [12, 13], [0, 14], [0, 15],
                       [14, 16], [15, 17]]
        n_points = 18

        return point_classes, point_pairs, n_points

    def mpi(self):
        point_classes = {
            "head": 0, "neck": 1,
            "right shoulder": 2, "right elbow": 3, "right wrist": 4,
            "left shoulder": 5, "left elbow": 6, "left wrist": 7,
            "right Hip": 8, "right knee": 9, "right ankle": 10,
            "left hip": 11, "left knee": 12, "left ankle": 13,
            "chest": 14, "background": 15
        }
        point_pairs = [[0, 1], [1, 2], [2, 3],
                       [3, 4], [1, 5], [5, 6],
                       [6, 7], [1, 14], [14, 8],
                       [8, 9], [9, 10], [14, 11],
                       [11, 12], [12, 13]
                       ]
        n_points = 15

        return point_classes, point_pairs, n_points

    def body25(self):
        point_pairs = [[1, 0], [1, 2], [1, 5],
                       [2, 3], [3, 4], [5, 6],
                       [6, 7], [0, 15], [15, 17],
                       [0, 16], [16, 18], [1, 8],
                       [8, 9], [9, 10], [10, 11],
                       [11, 22], [22, 23], [11, 24],
                       [8, 12], [12, 13], [13, 14],
                       [14, 19], [19, 20], [14, 21]]
        n_points = 25

        return point_pairs, n_points

    def start(self):
        cap = WebcamVideoStream(src=0).start()
        # cap = cv2.VideoCapture(0)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        while True:
            # camera = cv2.VideoCapture(0)
            img = cap.read()
            img_height, img_width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (self.in_width, self.in_height),
                                         (0, 0, 0), swapRB=False, crop=False)

            self.net.setInput(blob)

            output = self.net.forward()

            H = output.shape[2]
            W = output.shape[3]

            points = []

            for i in range(self.n_points):
                prob_map = output[0, i, :, :]

                min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

                x = (img_width * point[0]) / W
                y = (img_height * point[1]) / H

                if prob > 0.1:
                    cv2.circle(img, (int(x), int(y)), 15, (0, 255, 255),
                               1, cv2.FILLED)
                    cv2.putText(img, '%d' % i, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX,
                                1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            for pair in self.point_pairs:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(img,
                             points[partA],
                             points[partB],
                             (0, 255, 255), 3)
                    cv2.circle(img,
                               points[partA],
                               8,
                               (0, 0, 255),
                               thickness=-1,
                               lineType=cv2.FILLED)
                cv2.imshow('img', img)
                cv2.waitKey(1)


pose = pose_landmarks()
pose.start()
