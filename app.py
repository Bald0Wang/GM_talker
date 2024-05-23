import os
import re
import time
import shutil

import gradio as gr
import numpy as np

import os
import numpy as np
import cv2
import torch
import glob
import pickle
from tqdm import tqdm
import copy
from argparse import Namespace
import imageio
from moviepy.editor import *


ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")

def print_directory_contents(path):
    for child in os.listdir(path):
        child_path = os.path.join(path, child)
        if os.path.isdir(child_path):
            print(child_path)



from musetalk.utils.utils import get_file_type,get_video_fps,datagen
from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder,get_bbox_range
from musetalk.utils.blending import get_image
from musetalk.utils.utils import load_all_model





@torch.no_grad()
def inference(audio_path,video_path,bbox_shift,progress=gr.Progress(track_tqdm=True)):
    args_dict={"result_dir":'./results/output', "fps":25, "batch_size":8, "output_vid_name":'', "use_saved_coord":False}#same with inferenece script
    args = Namespace(**args_dict)

    input_basename = os.path.basename(video_path).split('.')[0]
    audio_basename  = os.path.basename(audio_path).split('.')[0]
    output_basename = f"{input_basename}_{audio_basename}"
    result_img_save_path = os.path.join(args.result_dir, output_basename) # related to video & audio inputs
    crop_coord_save_path = os.path.join(result_img_save_path, input_basename+".pkl") # only related to video input
    os.makedirs(result_img_save_path,exist_ok =True)

    if args.output_vid_name=="":
        output_vid_name = os.path.join(args.result_dir, output_basename+".mp4")
    else:
        output_vid_name = os.path.join(args.result_dir, args.output_vid_name)
    ############################################## extract frames from source video ##############################################
    if get_file_type(video_path)=="video":
        save_dir_full = os.path.join(args.result_dir, input_basename)
        os.makedirs(save_dir_full,exist_ok = True)
        # cmd = f"ffmpeg -v fatal -i {video_path} -start_number 0 {save_dir_full}/%08d.png"
        # os.system(cmd)
        # 读取视频
        reader = imageio.get_reader(video_path)

        # 保存图片
        for i, im in enumerate(reader):
            imageio.imwrite(f"{save_dir_full}/{i:08d}.png", im)
        input_img_list = sorted(glob.glob(os.path.join(save_dir_full, '*.[jpJP][pnPN]*[gG]')))
        fps = get_video_fps(video_path)
    else: # input img folder
        input_img_list = glob.glob(os.path.join(video_path, '*.[jpJP][pnPN]*[gG]'))
        input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        fps = args.fps
    #print(input_img_list)
    ############################################## extract audio feature ##############################################
    whisper_feature = audio_processor.audio2feat(audio_path)
    whisper_chunks = audio_processor.feature2chunks(feature_array=whisper_feature,fps=fps)
    ############################################## preprocess input image  ##############################################
    if os.path.exists(crop_coord_save_path) and args.use_saved_coord:
        print("using extracted coordinates")
        with open(crop_coord_save_path,'rb') as f:
            coord_list = pickle.load(f)
        frame_list = read_imgs(input_img_list)
    else:
        print("extracting landmarks...time consuming")
        coord_list, frame_list = get_landmark_and_bbox(input_img_list, bbox_shift)
        with open(crop_coord_save_path, 'wb') as f:
            pickle.dump(coord_list, f)
    bbox_shift_text=get_bbox_range(input_img_list, bbox_shift)
    i = 0
    input_latent_list = []
    for bbox, frame in zip(coord_list, frame_list):
        if bbox == coord_placeholder:
            continue
        x1, y1, x2, y2 = bbox
        crop_frame = frame[y1:y2, x1:x2]
        crop_frame = cv2.resize(crop_frame,(256,256),interpolation = cv2.INTER_LANCZOS4)
        latents = vae.get_latents_for_unet(crop_frame)
        input_latent_list.append(latents)

    # to smooth the first and the last frame
    frame_list_cycle = frame_list + frame_list[::-1]
    coord_list_cycle = coord_list + coord_list[::-1]
    input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
    ############################################## inference batch by batch ##############################################
    print("start inference")
    video_num = len(whisper_chunks)
    batch_size = args.batch_size
    gen = datagen(whisper_chunks,input_latent_list_cycle,batch_size)
    res_frame_list = []
    for i, (whisper_batch,latent_batch) in enumerate(tqdm(gen,total=int(np.ceil(float(video_num)/batch_size)))):
        
        tensor_list = [torch.FloatTensor(arr) for arr in whisper_batch]
        audio_feature_batch = torch.stack(tensor_list).to(unet.device) # torch, B, 5*N,384
        audio_feature_batch = pe(audio_feature_batch)
        
        pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
        recon = vae.decode_latents(pred_latents)
        for res_frame in recon:
            res_frame_list.append(res_frame)
            
    ############################################## pad to full image ##############################################
    print("pad talking image to original video")
    for i, res_frame in enumerate(tqdm(res_frame_list)):
        bbox = coord_list_cycle[i%(len(coord_list_cycle))]
        ori_frame = copy.deepcopy(frame_list_cycle[i%(len(frame_list_cycle))])
        x1, y1, x2, y2 = bbox
        try:
            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
        except:
    #                 print(bbox)
            continue
        
        combine_frame = get_image(ori_frame,res_frame,bbox)
        cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png",combine_frame)
        
    # cmd_img2video = f"ffmpeg -y -v fatal -r {fps} -f image2 -i {result_img_save_path}/%08d.png -vcodec libx264 -vf format=rgb24,scale=out_color_matrix=bt709,format=yuv420p temp.mp4"
    # print(cmd_img2video)
    # os.system(cmd_img2video)
    # 帧率
    fps = 25
    # 图片路径
    # 输出视频路径
    output_video = 'temp.mp4'

    # 读取图片
    def is_valid_image(file):
        pattern = re.compile(r'\d{8}\.png')
        return pattern.match(file)

    images = []
    files = [file for file in os.listdir(result_img_save_path) if is_valid_image(file)]
    files.sort(key=lambda x: int(x.split('.')[0]))

    for file in files:
        filename = os.path.join(result_img_save_path, file)
        images.append(imageio.imread(filename))
        

    # 保存视频
    imageio.mimwrite(output_video, images, 'FFMPEG', fps=fps, codec='libx264', pixelformat='yuv420p')

    # cmd_combine_audio = f"ffmpeg -y -v fatal -i {audio_path} -i temp.mp4 {output_vid_name}"
    # print(cmd_combine_audio)
    # os.system(cmd_combine_audio)

    input_video = './temp.mp4'
    # Check if the input_video and audio_path exist
    if not os.path.exists(input_video):
        raise FileNotFoundError(f"Input video file not found: {input_video}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # 读取视频
    reader = imageio.get_reader(input_video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

    # 将帧存储在列表中
    frames = images

    # 保存视频并添加音频
    # imageio.mimwrite(output_vid_name, frames, 'FFMPEG', fps=fps, codec='libx264', audio_codec='aac', input_params=['-i', audio_path])
    
    # input_video = ffmpeg.input(input_video)
    
    # input_audio = ffmpeg.input(audio_path)
    
    print(len(frames))

    # imageio.mimwrite(
    #     output_video,
    #     frames,
    #     'FFMPEG',
    #     fps=25,
    #     codec='libx264',
    #     audio_codec='aac',
    #     input_params=['-i', audio_path],
    #     output_params=['-y'],  # Add the '-y' flag to overwrite the output file if it exists
    # )
    # writer = imageio.get_writer(output_vid_name, fps = 25, codec='libx264', quality=10, pixelformat='yuvj444p')
    # for im in frames:
    #     writer.append_data(im)
    # writer.close()




    # Load the video
    video_clip = VideoFileClip(input_video)

    # Load the audio
    audio_clip = AudioFileClip(audio_path)

    # Set the audio to the video
    video_clip = video_clip.set_audio(audio_clip)

    # Write the output video
    video_clip.write_videofile(output_vid_name, codec='libx264', audio_codec='aac',fps=25)

    os.remove("temp.mp4")
    #shutil.rmtree(result_img_save_path)
    print(f"result is save to {output_vid_name}")
    return output_vid_name,bbox_shift_text



# load model weights
audio_processor,vae,unet,pe  = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)




def gfpgan_face_aug(audio_path, video_path, gfp_ver, sr_bg, progress=gr.Progress(track_tqdm=True)):
    # 创建目录
    tmp_root = '/root/autodl-tmp'
    tmp_dir_in = os.path.join(tmp_root, 'video_frames_in_xxx')
    tmp_dir_out = os.path.join(tmp_root, 'video_frames_out_xxx')
    for tmp_dir in [tmp_dir_in, tmp_dir_out]:
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
    
    old_work_dir = os.getcwd()
    os.chdir('/root/GFPGAN')
    
    # 生成输出文件名
    now = int(time.time())
    dirname, basename = os.path.split(video_path)
    filename, ext = os.path.splitext(basename)
    new_video_path = os.path.join('/root/MuseTalk/results/output', f'{filename}-gfpgan-{now}{ext}')

    # 获取视频帧率
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率
    reader.close()

    # 进度条开始
    progress(0, '开始修复 ...')
    pbar = progress.tqdm(range(4))

    # 拆帧
    next(pbar)
    cmd = f'ffmpeg -i "{video_path}" "{tmp_dir_in}/%06d.png"'
    print('拆帧: ', cmd)
    os.system(cmd)

    # 面部修复
    next(pbar)
    bg_upsampler = 'realesrgan' if sr_bg else 'none'
    cmd = f'python inference_gfpgan.py -v {gfp_ver} --bg_upsampler {bg_upsampler} -i "{tmp_dir_in}" -o "{tmp_dir_out}"'
    print('面部修复: ', cmd)
    os.system(cmd)

    # 合帧
    next(pbar)
    cmd = f'ffmpeg -r {fps} -i "{tmp_dir_out}/restored_imgs/%06d.png" -i "{audio_path}" -pix_fmt yuv420p {new_video_path}'
    print('合帧: ', cmd)
    os.system(cmd)

    next(pbar)
    try:
        shutil.rmtree(tmp_dir_in)
        shutil.rmtree(tmp_dir_out)
    except:
        pass
    os.chdir(old_work_dir)
    print('GFPGAN: done!')

    return new_video_path




def check_video(video):
    if not isinstance(video, str):
        return video # in case of none type
    # Define the output video file name
    dir_path, file_name = os.path.split(video)
    if file_name.startswith("outputxxx_"):
        return video
    # Add the output prefix to the file name
    output_file_name = "outputxxx_" + file_name

    os.makedirs('./results',exist_ok=True)
    os.makedirs('./results/output',exist_ok=True)
    os.makedirs('./results/input',exist_ok=True)

    # Combine the directory path and the new file name
    output_video = os.path.join('./results/input', output_file_name)


    # # Run the ffmpeg command to change the frame rate to 25fps
    # command = f"ffmpeg -i {video} -r 25 -vcodec libx264 -vtag hvc1 -pix_fmt yuv420p crf 18   {output_video}  -y"

    # 读取视频
    reader = imageio.get_reader(video)
    fps = reader.get_meta_data()['fps']  # 获取原视频的帧率

    # 将帧存储在列表中
    frames = [im for im in reader]

    # 保存视频
    imageio.mimwrite(output_video, frames, 'FFMPEG', fps=25, codec='libx264', quality=9, pixelformat='yuv420p')
    return output_video




with gr.Blocks() as demo:
    gr.Markdown("#MuseTalk: 实时唇形合成")

    with gr.Row():
        with gr.Column():
            audio = gr.Audio(label="驱动音频",type="filepath")
            video = gr.Video(label="参考视频",sources=['upload'])
            bbox_shift = gr.Number(label="BBox_shift 值, 单位：px", value=0)
            bbox_shift_scale = gr.Textbox(label="BBox_shift 建议值的下界，首次生成视频后，将会生成 bbox 范围。 \n 如果结果不好，可以根据这个参考值调整。", value="",interactive=False)
            btn_gen = gr.Button("生成")
        with gr.Column():
            out1 = gr.Video(label='MuseTalk合成的视频')
            gfp_ver = gr.Dropdown(choices=['1.3', '1.4'], value='1.3', label='GFPGAN 版本')
            sr_bg = gr.Checkbox(value=True, label='背景增强')
            btn_aug = gr.Button("GFPGAN面部高清修复")
            out2 = gr.Video(label='GFPGAN修复的视频')
    
    video.change(
        fn=check_video, inputs=[video], outputs=[video]
    )

    btn_gen.click(
        fn=inference,
        inputs=[
            audio,
            video,
            bbox_shift,
        ],
        outputs=[out1,bbox_shift_scale]
    )
    btn_aug.click(
        fn=gfpgan_face_aug,
        inputs=[
            audio,
            out1,
            gfp_ver,
            sr_bg,
        ],
        outputs=[out2]
    )

# Set the IP and port
ip_address = "0.0.0.0"  # Replace with your desired IP address
port_number = 6006  # Replace with your desired port number


demo.queue().launch(
    share=False , debug=True, server_name=ip_address, server_port=port_number
)
