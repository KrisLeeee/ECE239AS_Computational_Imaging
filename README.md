# ECE239AS_Computational_Imaging


### Project 1 : **Raw Image Data Analysis and Processing Assignment**

1. **Data Processing**
    - implement various image data processing functions  

![image](https://github.com/user-attachments/assets/7406616b-2b84-4fec-ba67-5a49c1639882)

a) Demosainking RAW Image
    - convert Bayer data into full RGB image
    - Bilinear Interpolation : each pixel values are averaged from its four neighbors
        -----------------
        |  red  | green |
        -----------------
        | green |  blue |
        -----------------
b) White Balancing
    - used gray-world assumption
c) Color Transformation
    - convert the image from XYZ to RGB color space. (matrix converting)
d) Gamma Encoding


2. **Denoising**
    - implement functions to estimate various noises in the imaging pipeline
    - estimate dark current and read adc

![image](https://github.com/user-attachments/assets/df86a39f-7688-4a7a-a8ed-91694af17f85)


3. **HDR Imaging**
Implement exprosure bracketing for HDR reconstruction.
With access to a set of images taken at various exposure levels, we output a single HDR image with better representation of the scene.


### Project 2 : **Creat 'Bokeh' Effect with All-in-focus Mobile Phone Video**

Overview : Synthesize images with smaller depths of field thus making it appear to have been taken from an expensive camera with large aperture.

1. Take an all-in focus video by moving the camera in zig-zag path, no tilt.
2. Use normalized cross correlation to extract pixel shift.
3. Synthesize a refocused ‘Bokeh’ by shifting each image in the opposite direction and sum up all frames.



### Project 3 : **NeRF Implementation**





### Project 4 : **Stable Diffusion**

explore a pre-trained stable diffusion model with parameter tuning. Try different VAE, text encoders, schedulers, etc.
