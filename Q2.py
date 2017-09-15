from PIL import Image
import sys
im = Image.open(sys.argv[1])
width = im.size[0]
height = im.size[1]
for w in range(0 , width):
    for h in range(0 , height):
        pixel = im.getpixel((w,h))
        new_pixel = (int(pixel[0]/2) , int(pixel[1]/2) , int(pixel[2]/2))
        im.putpixel([w , h] , new_pixel)

im.save('Q2.png')
