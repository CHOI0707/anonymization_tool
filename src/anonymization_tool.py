import numpy as np
import cv2
import os       # OS module 가져오기
import argparse # argument를 받아 처리할 수 있게 해줌.
import glob     # 여러 이미지를 한꺼번에 읽을 수 있게 해줌.
import time
import random
from DetectorAPI import Detector

def masking(image, boxes):

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        white = (255, 255, 255)

        # apply masking
        cv2.rectangle(image, (x1, y1), (x2, y2), white, thickness=cv2.FILLED)

    return image

def blurring(image, boxes, method, k):
    """
    Argument:
    image -- the image that will be edited as a matrix
    boxes -- list of boxes that will be blurred each element must be a dictionary that has [id, score, x1, y1, x2, y2] keys
    Returns:
    image -- the blurred image as a matrix
    """

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        # crop the image due to the current box
        sub = image[y1:y2, x1:x2]

        # apply blurring on cropped area
        if method == 1:
            blur = cv2.blur(sub, (k, k))
        elif method == 2:
            blur = cv2.GaussianBlur(sub, (k, k), 0)
        elif method == 3:
            blur = cv2.medianBlur(sub, k)
        else:
            print("Invalid input")
            exit()

        # paste blurred image on the original image
        image[y1:y2, x1:x2] = blur

    return image

def pixelating(image, boxes, k):

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        sub = image[y1:y2, x1:x2]
        h, w = sub.shape[:2]

        x_sampling = (x2-x1)//(2*k)
        y_sampling = (y2-y1)//(2*k)

        # The smallest is the resize, the biggest are the PIXELS
        imgSmall = cv2.resize(sub, (x_sampling, y_sampling))
        # Scale back up using NEAREST to original size
        pixel = cv2.resize(imgSmall, (w, h), interpolation=cv2.INTER_NEAREST)

        # apply pixelating
        image[y1:y2, x1:x2] = pixel

    return image

def motion_blurring(image, boxes, method, k):

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        sub = image[y1:y2, x1:x2]

        if method == 1:
            # Create the kernel
            kernel_v = np.zeros((k, k))
            # Fill the middle row with ones
            kernel_v[:, int((k - 1)/2)] = np.ones(k)
            # Normalize
            kernel_v /= k
            # Apply the vertical kernel
            vertical = cv2.filter2D(sub, -1, kernel_v)
            image[y1:y2,x1:x2] = vertical

        elif method == 2:
            kernel_h = np.zeros((k, k))
            kernel_h[int((k - 1)/2), :] = np.ones(k)
            kernel_h /= k
            horizontal = cv2.filter2D(sub, -1, kernel_h)
            image[y1:y2,x1:x2] = horizontal

        elif method == 3:
            kernel_d = np.zeros((k, k))
            np.fill_diagonal(kernel_d, 1)
            kernel_d /= k
            diagonal = cv2.filter2D(sub, -1, kernel_d)
            image[y1:y2,x1:x2] = diagonal

        elif method == 4:
            kernel_d = np.zeros((k, k))
            np.fill_diagonal(kernel_d, 1)
            kernel_rd = np.flipud(kernel_d)
            kernel_rd /= k
            r_diagonal = cv2.filter2D(sub, -1, kernel_rd)
            image[y1:y2,x1:x2] = r_diagonal

        else:
            print("Invalid input")
            exit()

    return image

def adding_pepper(image, boxes, method, k):

    for box in boxes:
        # unpack each box
        x1, y1 = box["x1"], box["y1"]
        x2, y2 = box["x2"], box["y2"]

        sub = image[y1:y2, x1:x2]

        h, w, c = sub.shape

        for i in range(h):
            row = 0
            while row < w-method:
                alpha = random.randrange(method)
                for j in range(k):
                    sub[i, row+alpha-j]=[0,0,0]
                row = row+alpha

        # apply random black pixels
        image[y1:y2, x1:x2] = sub

    return image



def main(args):
    # assign model path and threshold
    model_path = args.model_path
    threshold = args.threshold

    # create detection object
    detector = Detector(model_path=model_path, name="Face detector")

    # read all jpg images in folder
    img_files = glob.glob(args.input_image + '/*')

    count = len(img_files)
    index = 0

    while True:
        # open image
        img = cv2.imread(img_files[index])

        if img is None:
            print("There is an unreadable img file")
            exit()
        
        if index == 0:
            # if the size of the dataset is too small, anonymize entire image
            fd = 'y' == input("Do you need face detection? [y/n] -> ")

            # decide anonymization method
            print("\nChoose anonymization method:\n",
                              "1. Masking\n",
                              "2. Blurring\n",
                              "3. Pixelating\n",
                              "4. Motion blurring\n",
                              "5. Adding random black pixels")
            anony = int(input("-> "))

            if anony == 1:
                pass
            elif anony == 2:
                # decide blurring type
                print("Choose blurring type:\n",
                              "1. blur()\n",
                              "2. GaussianBlur()\n",
                              "3. medianBlur()")
                method = int(input("-> "))
                # decide filter size
                k = int(input("Filter size k (must be an odd number): "))
            elif anony == 3:
                # decide degree
                k = int(input("Pixelating level (1~15): "))
            elif anony == 4:
                # decide blurring type
                print("Choose motion blurring type:\n",
                              "1. vertically\n",
                              "2. horizontally\n",
                              "3. diagonally\n",
                              "4. reverse_diagonally")
                method = int(input("-> "))
                # decide filter size
                k = int(input("Filter size k: "))
            elif anony == 5:
                # decide level
                method = 23 - int(input("The amount of black pixels (1~20): "))
                # decide size
                k = int(input("Length of the black pixels (>= 1): "))
            else:
                print("Invalid input")
                exit()
        
        # face detection
        if fd:
            faces = detector.detect_objects(img, threshold=threshold)
        else:
            h, w, c = img.shape
            faces = [{'id':1, 'score':1, 'x1':0, 'y1':0, 'x2':w, 'y2':h}]

        # apply annonymization method
        if anony == 1:
            img = masking(img, faces)
        elif anony == 2:
            img = blurring(img, faces, method, k)
        elif anony == 3:
            img = pixelating(img, faces, k)
        elif anony == 4:
            img = motion_blurring(img, faces, method, k)
        elif anony == 5:
            img = adding_pepper(img, faces, method, k)

        if index == 0:
            # show first image
            cv2.imshow('anonymized image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            # estimate elapsed time
            start_time = time.time()
            print("\nAnonymizing all images in dataset may take a lot of time.")
        
        # save image
        cv2.imwrite(args.output_image + '/%d.jpg'%(index), img)

        # indicate progress
        index += 1
        msg = '\rProcessing... [%d/%d]'%(index, count)
        print(msg, end='')

        if index == count:
            end_time = time.time()
            print("\nAnonymization finised.")
            print("Elapsed time: {}".format(str(end_time - start_time)))
            break



if __name__ == "__main__":
    # creating argument parser
    parser = argparse.ArgumentParser(description='Image blurring parameters')

    # adding arguments
    parser.add_argument('-i',
                        '--input_image',
                        help='Path to your image',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to .pb model',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_image',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help='Face detection confidence',
                        default=0.7,
                        type=float)
    args = parser.parse_args()      # 입력받은 argument는 args에 저장됨.
    
    # if input directory is invalid then stop
    assert os.path.isdir(os.path.dirname(
            args.input_image)), 'No such directory'

    # if output directory is invalid then stop
    if args.output_image:
        assert os.path.isdir(os.path.dirname(
            args.output_image)), 'No such directory'

    main(args)
