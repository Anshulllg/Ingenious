# import numpy as np
# import cv2
# import mediapipe as mp
# from matplotlib import pyplot as plt
# import matplotlib.animation as animation
#
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
#
# # img_1 = np.zeros([512, 512, 1], dtype=np.uint8)
# # img_1.fill(255)
# # or img[:] = 255
# # cv2.imshow('Single Channel Window', img_1)
# # print("image shape: ", img_1.shape)
#
# img_3 = np.zeros([800, 800, 3], dtype=np.uint8)
# # img_3.fill(255)
#
# # cv2.imshow('3 Channel Window', img_3)
#
# x = 0
# y = 0
#
# # distance from camera to object(face) measured
# # centimeter
# Known_distance = 76.2
#
# # width of face in the real world or Object Plane
# # centimeter
# Known_width = 14.3
#
# # Colors
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# WHITE = (255, 255, 255)
# BLACK = (0, 0, 0)
#
# # defining the fonts
# fonts = cv2.FONT_HERSHEY_COMPLEX
#
# # face detector object
# face_detector = cv2.CascadeClassifier(
#     cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# # face_detector = cv2.data.haarcascades("haarcascade_frontalface_default.xml")
#
#
# def Focal_Length_Finder(measured_distance, real_width, width_in_rf_image):
#
#     # finding the focal length
#     focal_length = (width_in_rf_image * measured_distance) / real_width
#     return focal_length
#
# # distance estimation function
#
#
# def Distance_finder(Focal_Length, real_face_width, face_width_in_frame):
#
#     distance = (real_face_width * Focal_Length)/face_width_in_frame
#
#     # return the distance
#     return distance
#
#
# def face_data(image):
#
#     face_width = 0  # making face width to zero
#
#     # converting color image ot gray scale image
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # detecting face in the image
#     faces = face_detector.detectMultiScale(gray_image, 1.3, 5)
#
#     # looping through the faces detect in the image
#     # getting coordinates x, y , width and height
#     for (x, y, h, w) in faces:
#
#         # draw the rectangle on the face
#         cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)
#
#         # getting face width in the pixels
#         face_width = w
#
#     # return the face width in pixel
#     return face_width
#
#
# # reading reference_image from directory
# ref_image = cv2.imread("F:/air canvas/car5.jpg")
# # cv2.imshow("ref_img", ref_image)
# # find the face width(pixels) in the reference_image
# ref_image_face_width = face_data(ref_image)
#
# # get the focal by calling "Focal_Length_Finder"
# # face width in reference(pixels),
# # Known_distance(centimeters),
# # known_width(centimeters)
# Focal_length_found = Focal_Length_Finder(
#     Known_distance, Known_width, ref_image_face_width)
#
# print(Focal_length_found)
#
# Distance = 0
# prevPoint_x = 0
# prevPoint_y = 0
# # count = 0
#
#
# def fingerPoints(color):
#     count = 0
#     cap = cv2.VideoCapture(0)
#     with mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.2) as hands:
#
#         while cap.isOpened():
#
#             #*******************#
#
#             # reading the frame from camera
#             # _, frame = cap.read()
#
#             # # calling face_data function to find
#             # # the width of face(pixels) in the frame
#             # face_width_in_frame = face_data(frame)
#
#             # # check if the face is zero then not
#             # # find the distance
#             # if face_width_in_frame != 0:
#
#             #     # finding the distance by calling function
#             #     # Distance distance finder function need
#             #     # these arguments the Focal_Length,
#             #     # Known_width(centimeters),
#             #     # and Known_distance(centimeters)
#             #     Distance = Distance_finder(
#             #         Focal_length_found, Known_width, face_width_in_frame)
#
#             #     # draw line as background of text
#             #     # print(Distance)
#             #     cv2.line(frame, (30, 30), (230, 30), RED, 32)
#             #     cv2.line(frame, (30, 30), (230, 30), BLACK, 28)
#
#             #     # Drawing Text on the screen
#             #     cv2.putText(
#             #         frame, f"Distance: {round(Distance,2)} CM", (30, 35),
#             #         fonts, 0.6, GREEN, 2)
#
#             # # show the frame on the screen
#             # cv2.imshow("frame", frame)
#
#             # # quit the program if you press 'q' on keyboard
#             # if cv2.waitKey(1) == ord("q"):
#             #     break
#
#             # *********************
#             # if face_width_in_frame != 0:
#             #     Distance = Distance_finder(
#             #         Focal_length_found, Known_width, face_width_in_frame)
#
#             success, image = cap.read()
#             if not success:
#                 print("Ignoring empty camera frame.")
#                 # If loading a video, use 'break' instead of 'continue'.
#                 continue
#
#             # To improve performance, optionally mark the image as not writeable to
#             # pass by reference.
#             image.flags.writeable = False
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = hands.process(image)
#
#             image_height, image_width, _ = image.shape
#
#             # Draw the hand annotations on the image.
#             image.flags.writeable = True
#             image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     # print('hand_landmarks:', hand_landmarks)
#                     # print(
#                     #     f'Index finger tip coordinates: (',
#                     #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                     #     f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#                     # )
#                     # img_3[round(x)][round(y)][0] = 0
#                     # img_3[round(x)][round(y)][0] = 0
#                     # img_3[round(x)][round(y)][0] = 255
#                     cv2.imshow("draw", img_3)
#
#                     x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * 512
#                     y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * 512
#                     z = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z * 100
#                     print("z: ", z)
#                     if(z <= -10 and z >= -40):
#                         if(count != 0):
#                             cv2.line(img_3, (512-prevPoint_x, prevPoint_y),
#                                      (512-round(x), round(y)), color, 2)
#                         count = 1
#                         prevPoint_x = round(x)
#                         prevPoint_y = round(y)
#                     else:
#                         prevPoint_y = 0
#                         prevPoint_x = 0
#                         count = 0
#                     # else:
#                     #     cv2.line(img_3, (512-round(x), round(y)),
#                     #              (512-round(x), round(y)), [0, 0, 0], 6)
#                     mp_drawing.draw_landmarks(
#                         image,
#                         hand_landmarks,
#                         mp_hands.HAND_CONNECTIONS,
#                         mp_drawing_styles.get_default_hand_landmarks_style(),
#                         mp_drawing_styles.get_default_hand_connections_style())
#
#             # Flip the image horizontally for a selfie-view display.
#             # plt.show()
#             cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#             if cv2.waitKey(5) & 0xFF == 27:
#                 break
#
#     cap.release()
#
# #     width = int(cap.get(3))
# #     height = int(cap.get(4))
# # # Create a blank canvas
# #     canvas = np.zeros((height, width, 3), np.uint8)
#
#
# GREEN = (0, 255, 0)
# RED = (0, 0, 255)
# WHITE = (255, 255, 255)
# # BLACK = (0, 0, 0)
#
# print("GREEN: G")
# print("RED: R")
# print("WHITE: W")
# print("BLACK: B")
# color = input("Choose the color by their key words: ")
# if(color == 'G'):
#     fingerPoints(GREEN)
# if(color == 'R'):
#     fingerPoints(RED)
# if(color == 'W'):
#     fingerPoints(WHITE)