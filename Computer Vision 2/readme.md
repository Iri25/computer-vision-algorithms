### Problems:
1. Write an application that uses morphological operators to extract corners from an image using the
following algorithm:
- R1 = Dilate(Img,cross)
- R1 = Erode(R1,Diamond)
- R2 = Dilate(Img,Xshape)
- R2 = Erode(R2,square)
- R = absdiff(R2,R1)
- Display(R)
  
Transform the input image to make it compatible with binary operators and display the results imposed over the original image. 
