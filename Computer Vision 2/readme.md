### Problems:
Write an application that uses morphological operators to extract corners from an image using the
following algorithm:
1. R1 = Dilate(Img,cross)
2. R1 = Erode(R1,Diamond)
3. R2 = Dilate(Img,Xshape)
4. R2 = Erode(R2,square)
5. R = absdiff(R2,R1)
6. Display(R)
  
Transform the input image to make it compatible with binary operators and display the results imposed over the original image. 
