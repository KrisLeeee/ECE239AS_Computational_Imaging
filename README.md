# ECE239AS_Computational_Imaging


## Project 1 : **Raw Image Data Analysis and Processing Assignment**

### 1. **Data Processing**
    - implement various image data processing functions  

![image](https://github.com/user-attachments/assets/7406616b-2b84-4fec-ba67-5a49c1639882)

<img width="571" height="228" alt="image" src="https://github.com/user-attachments/assets/73c78ab5-2df0-494d-87c8-0a1e3aaff77a" />


#### a) Demosaiking RAW Image  
  - convert Bayer data into full RGB image  
  - Bilinear Interpolation : each pixel values are averaged from its four neighbors  
    - bayer_img = [H, W] numpy array with 2x2 mosaic pattern [[r, g], [g, b]]  
    - rgb_img = [H, W, 3] numpy array  
    - use scipy.signal.convolve2d for interpolation
    - ```
      # green interpolation
      g_filter = np.array([[0,1/4,0],[1/4,1,1/4],[0,1/4,0]])
      g = scipy.signal.convolve2d(g, g_filter, 'same', boundary='wrap')

      # red interpolation
      rb_filter = np.array([[1/4,1/2,1/4],[1/2,1,1/2],[1/4,1/2,1/4]])
      r = scipy.signal.convolve2d(r, rb_filter, 'same', boundary='wrap')

      # blue interpolation
      b = scipy.signal.convolve2d(b, rb_filter, 'same', boundary='wrap')
      ```


#### b) White Balancing
  - adjusts colors so that white appears neutral under different lighting conditions
  - used gray-world assumption
  - ```
    avg = np.mean(demosaicked_image, axis=(0,1))    # (avg_r, avg_g, avg_b)
    scale = avg[1]/avg                              # (avg_g/avg_r, 1, avg_g/avg_b) = (alpha,1,beta)
    whitebalance = demosaicked_image * scale        # (alpha*r, g, beta,b)
    ```
    
#### c) Color Transformation
  - convert the image from XYZ to RGB color space. (matrix converting)
    - find rgb to xyz color space matrix from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    - load xyz to camera color space matrix from picture metadata
    - matrix multiplication : xyz_to_rgb * cam_to_xyz * img.T
    - `rgb_img = np.reshape(rgb.T, (img.shape))`

#### d) Gamma Encoding
  - enhance visual quality by matching brightness representation to human perception, especially in dark areas
  - `([0,1] clipped image)^γ`. common choice `γ = 1/2.2`



### 2. **Denoising**
    - implement functions to estimate various noises in the imaging pipeline
    - estimate dark current and read adc

![image](https://github.com/user-attachments/assets/df86a39f-7688-4a7a-a8ed-91694af17f85)


estimate dark current, scene flux, read noise, and adc noise.

$$n_{add} = \mathcal{N}(0, \sqrt{\sigma_{read}^2 \cdot g^2 + \sigma_{adc}^2})$$



### 3. **HDR Imaging**
Implement exprosure bracketing for HDR reconstruction.
With access to a set of images taken at various exposure levels, we output a single HDR image with better representation of the scene.

<img width="588" height="189" alt="image" src="https://github.com/user-attachments/assets/b7eef660-5645-4e8d-8dec-ddf5963a2871" />

<img width="300" height="414" alt="image" src="https://github.com/user-attachments/assets/31edce28-8a1f-4fb5-96f4-a6d80dc1c500" />

  - if 0.05 pixel < pixel value < 0.95 then valid
  - append valid pixel value then weight by exposure
  - compute weighted average
    - `weights[valid] = 1 / (np.abs(np.log2(exposure))+1.0)`





## Project 2 : **Creat 'Bokeh' Effect with All-in-focus Mobile Phone Video**

Overview : Synthesize images with smaller depths of field thus making it appear to have been taken from an expensive camera with large aperture.


### 1. Take an all-in focus video by moving the camera in zig-zag path, no tilt.

<img width="979" height="208" alt="image" src="https://github.com/user-attachments/assets/92344f58-3f67-46c2-988d-ee4b054cc51e" />


### 2. Use normalized cross correlation to extract pixel shift.

  - normalized cross-correlation coefficient : $A[i, j]$
  - mean of template : $\bar{t}$
  - mean of the window $w[n,m]$ in the region under the template : $\bar{w}_{i,j}$
  - A[i,j] = sum[(w(n,m)-w̄_{i,j})(t(n-i,m-j)-t̄)] / sqrt( sum[(w(n,m)-w̄_{i,j})² (t(n-i,m-j)-t̄)²] )
  - `res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)`

<img width="560" height="442" alt="image" src="https://github.com/user-attachments/assets/3aa2d95d-2ed6-48e8-b2db-d48319352b29" />


### 3. Synthesize a refocused ‘Bokeh’ by shifting each image in the opposite direction and sum up all frames.

  - pixel shift vector for Frame Image $I_i[n,m]$ : $[s_{xi},s_{yi}]$
  - image output : $P[n,m]$
  - $$P[n,m] = \sum_{i}{I_i[n-s_{xi},m-s_{yi}]}$$
  - ```
    def compute_defocused_image(frames_color, shifts):
        #**STUDENT TODO**
        shift_sum = np.zeros_like(frames_color[0], dtype=np.float32)
        aff = np.array([[1,0,0],[0,1,0]], dtype=np.float32)
        seq_len = len(frames_color)
        for i in range(seq_len):
            aff[1,2] = shifts[0,0] - shifts[i,0]    # y
            aff[0,2] = shifts[0,1] - shifts[i,1]    # x
            shift_sum += cv2.warpAffine(frames_color[i],aff,(0,0))
        defocused = shift_sum / seq_len
        return defocused.astype(np.uint8)
    ```


<img width="770" height="419" alt="image" src="https://github.com/user-attachments/assets/3e0337b5-6fe4-480e-8096-c9be008857f3" />
<img width="770" height="419" alt="image" src="https://github.com/user-attachments/assets/e1b62a9e-5098-4d18-a77b-44836dd108ee" />






## Project 3 : **NeRF Implementation**


<img width="1172" height="343" alt="image" src="https://github.com/user-attachments/assets/cc5e1862-1774-4b00-82f4-c7be5c48db08" />

Neural Radiance Fields (NeRF) is a method that uses a neural network to create realistic 3D views of a scene from 2D images. By learning how light behaves at different points and angles, it can generate new views from camera positions that weren’t in the original input.


### Compute World Rays

<img width="928" height="252" alt="image" src="https://github.com/user-attachments/assets/e4958002-b9c9-4d29-a694-19e32de05dbf" />

  - compute grid for x, y components : `torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy')`
  - compute ray direction : use (x, y) index, image shape, focal length
    - ```
      x = (i-W/2)/focal           # i++ -> x++; W/2 = 0
      y = -(j-H/2)/focal          # j++ -> y--; H/2 = 0
      z = -torch.ones_like(x)     # f:1 = i:x
      dirs = torch.stack([x,y,z], dim=-1)
      ```
  - use ray direction to compute world rays (world origin and world direction)
    - requires camera space to world space matrix : shape (4, 4)
    - c2w matrix is unique to each pose an image is taken in
    - `world_ray_origin = torch.ones_like(dirs) * c2w[:3,3]  # column vector at rightmost column`
    - `world_ray_direction = dirs @ torch.t(c2w[:3,:3])      # (c2w[:,:3] @ dirs.T).T ; dirs is given as a row vector`

### Rendering Rays (Painting) : 

#### 1.   Compute 3D query points we want to paint/render in our scene
  - calculate where along the world rays we should check the scene's color and density
  - sample set of linearly distanced points between the factors of 'near' and 'far' for each ray (used for neural network to find color and density)
  - ```
    def student_compute_query_points(rays_o, rays_d, near, far, N_samples, rand=False):
        noise_std = 1e-3
        z_vals = torch.linspace(near, far, steps=N_samples)
        z_vals = z_vals + noise_std * torch.rand(z_vals.shape) if rand else z_vals # [N_samples]: distance along an arbitrary ray to each query point
        pts = world_ray_origin[...,None,:] + z_vals[...,:,None] * world_ray_direction[...,None,:] # [H,W,N_samples,3]: 3D query points
    
        return pts, z_vals
    ```
#### 2.   Feed Query Points to Neural Network to determine the color and density at those points
  - feed the queried 3D points into neural network(painter)
  - network will return the color and density for each point given
  - **positional encoding** : modify inputs for neural network (neural network cannot understand the inputs directly from the 3D coordinates)
    - Input:
      - x : input coordinates [B,3]
      - L : number of layers
    - Output:
      - pos_enc : Positional encoding of shape [B,3+(3*2*L)]

  - feed the points into the network and return the colors and opacities.

#### 3.   Find the weights associated with the density of each position
  - Weight : quantify the opaqueness for each points. foggy/opaque spot affect the ray's color a lot, while clear, doesn't so much.
    - remove any opacities that are less than 0
    - calculate the distances between each point we queried on each ray
    - `transparency = 1 - exp(-density * thickness)`
    - `weights = transparency * cumulative_product(1 - transparency, exclusive=True)   # [H,W,N_samples]`
      - opaque segments early in the rays contribute to the image's final color more than segments later in the rays

#### 4.   Blend the colors using the weights to to get the final color for each ray (paint/render our scene)
  - Using the past three functions, we will paint/render our rays to produce an image
  - ```
    rgb_map = torch.sum(weights[...,:3] * rgb, -2)  # [H,W,3]
    depth_map = torch.sum(weights[...,3] * z_vals, -1) # z locations * weights [H,W]
    acc_map = torch.sum(weights[...,3], -1) # all sum weight.   [H,W]
    ```




### Neural Network NeRF Model
  - 8 intermediate layers (width 256) with ReLU's after each
  - 4 output channels
  - skip connection every 4 layers
  - optimizer : Adam


<img width="811" height="418" alt="image" src="https://github.com/user-attachments/assets/738f1571-a33b-4920-baa8-ebe9243a86af" />

<img width="811" height="419" alt="image" src="https://github.com/user-attachments/assets/d50fe494-5f2b-4072-9cc5-0a9335cc9497" />

<img width="823" height="458" alt="image" src="https://github.com/user-attachments/assets/b1fd8b9c-8d83-4da9-b6bf-daf3f85f5cdf" />


## Project 4 : **Stable Diffusion**

explore a pre-trained stable diffusion model with parameter tuning. Try different VAE, text encoders, schedulers, etc.

### Pipeline

load the pre-trained weights of all components of the model. In this notebook we use Stable Diffusion version 1.4 ([CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)).

```
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

```







