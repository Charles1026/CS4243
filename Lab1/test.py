import lab1
import numpy as np



if __name__ == "__main__":
  #Test pad
  print("grayscale pad test")
  list = [[1]]
  img = np.asarray(list)
  print(lab1.pad_zeros(img, 1, 2, 1, 2))
  
  print("grayscale pad test large")
  list = [[1,2,3],[4,5,6],[7,8,9]]
  img = np.asarray(list)
  print(lab1.pad_zeros(img, 0, 10, 10, 2))
  
  print("color pad test")
  list = [[[255,0,123]]]
  img = np.asarray(list)
  print(lab1.pad_zeros(img, 1, 2, 1, 2))
  
  print("rgb 2 gray test")
  list = [[[255,0,123]]]
  img = np.asarray(list)
  print(lab1.rgb2gray(img))
  
  print("rgb 2 gray test large")
  list = [[[255,0,123], [0,0,0], [255,255,255]], [[255,123,123], [0,2,0], [255,234,255]]]
  img = np.asarray(list)
  print(lab1.rgb2gray(img))
  
  print("\ngray 2 grad test")
  list = [[1,2,3],[4,5,6],[7,8,9]]
  inputImg = np.asarray(list, dtype = float)
  imgVec = lab1.gray2grad(inputImg)
  for img in imgVec:
    print(img)
  
  print("\ngray 2 grad test Large")
  list = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20],[21,22,23,24,25]]
  inputImg = np.asarray(list, dtype = float)
  imgVec = lab1.gray2grad(inputImg)
  for img in imgVec:
    print(img)