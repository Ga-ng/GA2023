from mmseg.apis import init_model, inference_model, show_result_pyplot
path = 'mmsegmentation-main/'
config_path = path + 'configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_path = path + 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
img_path = 'demo/demo.png'


# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
result = inference_model(model, img_path)

# display the segmentation result
vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# Modify the time of displaying images, note that 0 is the special value that means "forever"
vis_image = show_result_pyplot(model, img_path, result, wait_time=5)