# CycleGAN
Synthesizing Computer Tomographic Vertebrae Medical Images using CycleGAN. 

### CycleGAN Original Project
![image](https://github.com/user-attachments/assets/668baf08-79d3-4850-a118-73cd34543dad)
### CycleGAN Medical Image Synthesis
![image](https://github.com/user-attachments/assets/ad07836c-14e6-4fb4-9ab8-c8c25aef77cd)

# Prerequisites
<pre>
numpy~=1.25.2
scipy~=1.11.1
imageio~=2.26.0
tqdm~=4.65.0
torchvision~=0.15.2
tifffile~=2021.7.2
</pre>
# Train
<pre>
python train.py --dataset_dir=med_image
</pre>
# Test
<pre>
python test.py --dataset_dir=med-image 
</pre>
# References
<pre>
CycleGAN, 
paper: Zhu, J. Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).
URL: [https://github.com/junyanz/CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix.git)
</pre>
