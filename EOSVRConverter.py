import numpy as np
import cv2
import sys
from subprocess import Popen
from multiprocessing import Pool
from subprocess import Popen, STDOUT, PIPE
from tqdm import tqdm
from os import mkdir, listdir, cpu_count
import os
from os.path import exists, join
from glob import glob
import pickle
from enum import Enum
from pathlib import Path
import lut3d
from functools import partial
import numpy as np
from pillow_lut import load_cube_file

# download canon lut from  (bottom of the page)
# https://tools.rodrigopolo.com/canonluts/ 

TOPAZ_BIN = r"C:\Program Files\Topaz Labs LLC\Topaz Sharpen AI\Topaz Sharpen AI.exe"

# Code adapted from https://gist.github.com/HViktorTsoi/8e8b0468a9fb07842669aa368382a7df
# Did some changes to speed up by 33% while nearly not changed the result
def shadowHighlightSaturationAdjustment(
        img,
        shadow_amount_percent, shadow_tone_percent, shadow_radius,
        highlight_amount_percent, highlight_tone_percent, highlight_radius,
        color_percent
):
    """
    Image Shadow / Highlight Correction. The same function as it in Photoshop / GIMP
    :param img: input RGB image numpy array of shape (height, width, 3)
    :param shadow_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param shadow_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param shadow_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param highlight_amount_percent [0.0 ~ 1.0]: Controls (separately for the highlight and shadow values in the image) how much of a correction to make.
    :param highlight_tone_percent [0.0 ~ 1.0]: Controls the range of tones in the shadows or highlights that are modified.
    :param highlight_radius [>0]: Controls the size of the local neighborhood around each pixel
    :param color_percent [-1.0 ~ 1.0]:
    :return:
    """
    shadow_tone = shadow_tone_percent * 255
    highlight_tone = 255 - highlight_tone_percent * 255

    shadow_gain = 1 + shadow_amount_percent * 6
    highlight_gain = 1 + highlight_amount_percent * 6

    # The entire correction process is carried out in YUV space,
    # adjust highlights/shadows in Y space, and adjust colors in UV space
    # convert to Y channel (grey intensity) and UV channel (color)
    height, width = img.shape[:2]
    imgYUV = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype('float32')
    img_Y, img_U, img_V = imgYUV[..., 0].reshape(-1), imgYUV[..., 1].reshape(-1), imgYUV[..., 2].reshape(-1)
    img_U -= 127
    img_V -= 127

    # extract shadow / highlight
    shadow_map = 255 - img_Y * 255 / shadow_tone
    shadow_map[np.where(img_Y >= shadow_tone)] = 0
    highlight_map = 255 - (255 - img_Y) * 255 / (255 - highlight_tone)
    highlight_map[np.where(img_Y <= highlight_tone)] = 0

    # // Gaussian blur on tone map, for smoother transition
    if shadow_amount_percent * shadow_radius > 0:
        # shadow_map = cv2.GaussianBlur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius), sigmaX=0).reshape(-1)
        shadow_map = cv2.blur(shadow_map.reshape(height, width), ksize=(shadow_radius, shadow_radius)).reshape(-1)

    if highlight_amount_percent * highlight_radius > 0:
        # highlight_map = cv2.GaussianBlur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius), sigmaX=0).reshape(-1)
        highlight_map = cv2.blur(highlight_map.reshape(height, width), ksize=(highlight_radius, highlight_radius)).reshape(-1)

    # Tone LUT
    t = np.arange(256)
    LUT_shadow = (1 - np.power(1 - t * (1 / 255), shadow_gain)) * 255
    LUT_shadow = np.maximum(0, np.minimum(255, np.int_(LUT_shadow + .5)))
    LUT_highlight = np.power(t * (1 / 255), highlight_gain) * 255
    LUT_highlight = np.maximum(0, np.minimum(255, np.int_(LUT_highlight + .5)))

    # adjust tone
    shadow_map = shadow_map * (1 / 255)
    highlight_map = highlight_map * (1 / 255)

    iH = (1 - shadow_map) * img_Y + shadow_map * LUT_shadow[np.int_(img_Y)]
    iH = (1 - highlight_map) * iH + highlight_map * LUT_highlight[np.int_(iH)]
    img_Y = iH

    # adjust color
    if color_percent != 0:
        # color LUT
        if color_percent > 0:
            LUT = (1 - np.sqrt(np.arange(32768)) * (1 / 128)) * color_percent + 1
        else:
            LUT = np.sqrt(np.arange(32768)) * (1 / 128) * color_percent + 1

        # adjust color saturation adaptively according to highlights/shadows
        color_gain = LUT[np.int_(img_U ** 2 + img_V ** 2 + .5)]
        w = 1 - np.minimum(2 - (shadow_map + highlight_map), 1)
        img_U = w * img_U + (1 - w) * img_U * color_gain
        img_V = w * img_V + (1 - w) * img_V * color_gain

    # re convert to RGB channel
    img_Y = img_Y.astype('uint8')
    img_U = (img_U + 127).astype('uint8')
    img_V = (img_V + 127).astype('uint8')
    imgYUV = np.row_stack([img_Y, img_U, img_V]).T.reshape(height, width, 3)
    output = cv2.cvtColor(imgYUV, cv2.COLOR_YUV2BGR)
    return output


# From https://github.com/kylemcdonald/FisheyeToEquirectangular
class FisheyeToEquirectangular:
    def __init__(self, n=4096, side=3600, blending=0, aperture=1):
        self.blending = blending
        blending_ratio = blending / n

        npy_file = f'fisheye-{n}-{side}.npy'

        if exists(npy_file):
            data = np.load(npy_file)
            self.x, self.y = data
        else:
            x_samples = np.linspace(0-blending_ratio, 1+blending_ratio, n+blending*2)
            y_samples = np.linspace(-1, 1, n)

            # equirectangular
            x, y = np.meshgrid(x_samples, y_samples)

            # longitude/latitude
            longitude = x * np.pi
            latitude = y * np.pi / 2

            # 3d vector
            Px = np.cos(latitude) * np.cos(longitude)
            Py = np.cos(latitude) * np.sin(longitude)
            Pz = np.sin(latitude)

            # 2d fisheye
            aperture *= np.pi
            r = 2 * np.arctan2(np.sqrt(Px*Px + Pz*Pz), Py) / aperture
            theta = np.arctan2(Pz, Px)
            theta += np.pi
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            x = np.clip(x, -1, 1)
            y = np.clip(y, -1, 1)

            x = (-x + 1) * side / 2
            y = (y + 1) * side / 2

            self.x = x.astype(np.float32)
            self.y = y.astype(np.float32)
            data = [self.x, self.y]
            np.save(npy_file, data)
    
    def unwarp_single(self, img, interpolation=cv2.INTER_LINEAR, border=cv2.BORDER_REFLECT):
        return cv2.remap(
            img, self.x, self.y,
            interpolation=interpolation,
            borderMode=border
        )

    def getLeftRightFisheyeImage(self, imgfn):
        try:
            img = cv2.imread(imgfn)
            h, w, c = img.shape
            global args
            # print (f"--- getLeftRightFisheyeImage {h} {w} {c} {args.side} ---")
            centerLx = w // 4 - 50
            centerRx = w * 3 // 4 + 50
            centerY = h // 2 + 50
            fisheyeR = int(args.side) // 2

            ymr = centerY - fisheyeR
            ypr = centerY + fisheyeR
            lmr = centerLx - fisheyeR
            lpr = centerLx + fisheyeR
            rmr = centerRx - fisheyeR
            rpr = centerRx + fisheyeR

            # print (f"--- getLeftRightFisheyeImage  {ymr}  {ypr}  {lmr}  {lpr}  {rmr}  {rpr} ---")
            imgL = img[ymr: ypr, lmr: lpr, :]
            imgR = img[ymr: ypr, rmr: rpr, :]
            return imgL, imgR
        except Exception as e:
            print (f"--- Error in correctForImage {imgfn} {e} {img.shape} ---")

    def correctForImage(self, imgfn, outfn, shallAdjust=False):
        try:
            # print (f"--- correctForImage {imgfn} {outfn} ---")
            imgL, imgR = self.getLeftRightFisheyeImage(imgfn)
            # print (f"--- imgL {imgL.shape} ---")
            newimg = self.unwarp_single(imgL)
            newimgR = cv2.rotate(newimg, cv2.ROTATE_180)
            newimg = self.unwarp_single(imgR)
            newimgL = cv2.rotate(newimg, cv2.ROTATE_180)
            newimg = np.hstack((newimgL, newimgR))
            if shallAdjust:
                # Perform some image processing
                newimg = shadowHighlightSaturationAdjustment(newimg, 0.05, 0.4, 50, 0.1, 0.4, 50, 0.4)
            cv2.imwrite(outfn, newimg)
        except Exception as e:
            print (f"--- Error in correctForImage {imgfn} {outfn} {e} ---")

    # Correct all images under the correct directory in place
    def correctAllImages(self, pool):
        fns = glob('*.jpg')
        
        pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))
        # Use Topaz Sharpen AI to enhance resolution
        command = [TOPAZ_BIN] + fns
        process = Popen(command)
        process.wait()

    # Extract frames from the video using ffmpeg, and then perform correction for each frame (in place)
    # Note the video here could be exported from Premiere or other software, and not necessarily the
    # out-of-body mp4 files. So even RAW could be supported (indirectly).
    # We also give another example of directly reading in the video file (not RAW though, could be ALL-I)
    # before color grading, and invoke ffmpeg to do color grading.
    def correctForVideo(self, videofn, pool, args):
        videofn_stem =Path(videofn).stem
        video_outdir = os.path.dirname(videofn)

        frames_outdir = f"{videofn_stem}_FRAMES"

        if not exists(frames_outdir):
            mkdir(frames_outdir)
        
        log_outfile_mp4_to_pngs = os.path.join(video_outdir, f"{videofn_stem}-mp4_to_pngs-log.txt")
        png_outdir = os.path.join(frames_outdir, "%05d.png")
        if (args.mp4_to_pngs):
            print ("--- mp4_to_pngs ---")
            # Example 1: don't do color grading
            ffmpegCommand = ['ffmpeg', '-i', videofn, '-qscale:v', '2']
            if (args.frames_v != None):
                ffmpegCommand += ['-frames:v', str(args.frames_v)]
            ffmpegCommand += [png_outdir]
            print (f"ffmpegCommand: {' '.join(ffmpegCommand)}")
            # Example 2: do color grading. Change the cube file path to your case.
            # Cube files can be downloaded from Canon website.
            # ffmpegCommand = ['ffmpeg', '-i', videofn, '-qscale:v', '2', '-vf', 'lut3d=BT2020_CanonLog3-to-BT709_WideDR_33_FF_Ver.2.0.cube', png_outdir]
            with open(log_outfile_mp4_to_pngs, 'w+') as f:
                exe = Popen(ffmpegCommand, stdout=PIPE, stderr=STDOUT, universal_newlines=True, text=True)
                for line in exe.stdout:
                    # line = line.decode("utf-8")
                    #   Duration: 00:00:41.58, start: 0.000000, bitrate: 439623 kb/s
                    if line.lstrip().startswith('Duration:'):
                        duration = line.lstrip().removeprefix('Duration:').strip().split(' ')[0].rstrip(',')
                        print (f"duration: {duration}")
                        # print (f"  '{line}'")
                    #   Stream #0:0[0x1](und): Video: hevc (Rext) (hvc1 / 0x31637668), yuv420p12le(tv, bt709), 4096x2160 [SAR 1:1 DAR 256:135], 438659 kb/s, 59.94 fps, 59.94 tbr, 60k tbn (default)
                    if (line.lstrip().startswith('Stream')) and ('fps,' in line):
                        fps = line.split(' fps,')[0].split(' ')[-1]
                        print (f"fps: {fps}")
                        # print (f"'{line}'    '{line.split(' fps,')[-1]}'")
                    # frame=    0 fps=0.0 q=0.0 size=       0KiB time=N/A bitrate=N/A speed=N/A    
                    # frame=   12 fps=3.0 q=-0.0 size=N/A time=00:00:00.20 bitrate=N/A speed=0.0494x    
                    if line.strip().startswith('frame='):
                        framenum = line.strip().removeprefix('frame=').strip().split(' ')[0]
                        # print (f'X', end='', flush=True)
                        # sys.stdout.flush()
                        print (f"{framenum}")
                        # print (f" '{line}'")
                    # print (line, end="")
                    # sys.stdout.flush()
                    f.write(line)
                exe.wait()
                
        if (args.mp4_to_pngs_opencv):
            print ("--- mp4_to_pngs_opencv ---")
            cam = cv2.VideoCapture(str(videofn))
            # has no influence, but valid
            # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'h264')) # h264, mp4v, MJPG
            # stops working
            # cam.set(cv2.IMREAD_ANYDEPTH, 1)
            # no influence, but valid
            # cam.set(cv2.CAP_PROP_FORMAT, cv2.CV_32FC1)
            if not cam.isOpened():
                print ("Error when reading image file")

            lut = None
            if (args.lut3d is not None):
                lut = load_cube_file(args.lut3d)
                # lut = lut3d.read_lut_file(args.lut3d)

            print (f"--- length {cam.get(cv2.CAP_PROP_FRAME_COUNT)} ---")
            index = 1
            while True:
                ret, frame = cam.read(cv2.IMREAD_ANYDEPTH)
                if frame is None:
                    break
                if lut is not None:
                    # pixels = frame.reshape(-1, 3)
                    # convert = partial(lut3d._convert, lut=lut)
                    # new_pixels = list(map(convert, tqdm(pixels)))
                    # frame = np.array(new_pixels).reshape(frame.shape)
                    # frame = (frame * 255).astype('uint8')

                    frame = cv2.LUT(frame, lut)
                    
                    calibrate = cv.createCalibrateDebevec()
                    response = calibrate.process(frame, times)

                    # pil image filter
                    # im.filter(lut).save("image-with-lut-applied.png")

                cv2.imwrite(png_outdir % index, frame)
                index += 1
                print ("X", end="", flush=True)
                # cv2.LUT(frame, cv2.COLORMAP_JET, frame)
            cam.release()
                
        # Also extract the audio to be combined to the final video
        log_outfile_mp4_to_aac = os.path.join(video_outdir, f"{videofn_stem}-mp4_to_aac-log.txt")
        audio_out_fname = f'{videofn_stem}_audio.aac'
        if (args.mp4_to_aac):
            print ("--- mp4_to_aac ---")
            ffmpegAudioCommand = ['ffmpeg', '-y', '-i', videofn, '-vn', '-acodec', 'copy', audio_out_fname]
            with open(log_outfile_mp4_to_aac, 'w+') as f:
                exe = Popen(ffmpegAudioCommand, stdout=f, stderr=STDOUT)
                exe.wait()

        # Perform the mapping and adjustment (slow) in parallel
        if (args.png_unwrap):
            print ("--- png_unwrap ---")
            fns = [os.path.join(frames_outdir, x) for x in listdir(frames_outdir)]
            #pool.starmap(self.correctForImage, tqdm([(fn, fn) for fn in fns]))
            pool.starmap(self.correctForImage, tqdm([(fn, fn, True) for fn in fns]))
        
        log_outfile_png_to_mp4 = os.path.join(video_outdir, f"{videofn_stem}-png_to_mp4-log.txt")
        video_out_fname = f'{videofn_stem}_VR.mp4'
        if (args.png_to_mp4):
            print ("--- png_to_mp4 ---")
            # Get an initial version without sharpening for quick review
            command = ['ffmpeg', '-r', '30', '-i', png_outdir, '-i', audio_out_fname, '-c:v', 'libx264', '-vf', 'scale=8192x4096', '-preset', 'fast', '-crf', '18', '-x264-params', 'mvrange=511', '-maxrate', '100M', '-bufsize', '25M', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '160k', '-movflags', 'faststart', video_out_fname]
            with open(log_outfile_png_to_mp4, 'w+') as f:
                exe = Popen(command, stdout=f, stderr=STDOUT)
                exe.wait()

    def correctAllVideos(self, args):
        fns = glob('*.mp4')
        for fn in fns:
            print(f'Processing {fn}...')
            self.correctForVideo(fn, f'{fn.replace(".mp4", "")}_Frames', args)



class Args:
    mp4_to_pngs = True
    # is bad, can not handle hdr images
    mp4_to_pngs_opencv = True
    mp4_to_aac = True
    png_unwrap = True
    png_to_mp4 = True
    test_conversion = False
    # must be 4096 for images with 4096 x 2160 or 8192 for 8192 x 4320
    n = 4096
    # must be 1800 for images with 4096 x 2160 or 3600 for 8192 x 4320
    side = 3600

    frames_v = None
    # extremely slow!!! 20sec / frame!!!
    lut3d = None

# overwrite for 4k video resolution
args = Args()
args.n = 2048
args.side = 1800
# args.lut3d = "BT2020_CanonLog3-to-BT709_WideDR_33_FF_Ver.2.0.cube"
# process only 100 frames
#args.frames_v = 100

if __name__ == '__main__':
    # We don't have a command line interface for now to provide maximum 
    # efficiency (e.g. no need to intialize FisheyeToEquirectangular every time)
    # Sample usage:

    procCount = min(56, cpu_count() // 3 * 2)
    print(f'Creating a process pool with {procCount} processed...')
    pool = Pool(procCount)

    # prepare converter
    converter = FisheyeToEquirectangular(n=args.n, side=args.side)
    
    if (args.test_conversion):
        converter.correctForImage('./A001C002_240520L4_CANON_FRAMES/00071.png', './00071_corrected.png')
        #converter.correctForImage('./tmpframes2/00001.png', './tmpframes2/00001_corrected.png')
        exit (0)
    
    converter.correctForVideo(
        '/Users/rspoerri/_REPOSITORIES/_EXTERNAL/EOSVRConverter/A001C002_240520L4_CANON.mp4',
        pool=pool, args=args)

    #converter.correctAllImages(args)
    #for i in list(reversed(range(4014, 4019))):
        #fn = f'IMG_{i}.MP4'
        #newfn = f'IMG_{i}'
        #converter.correctForVideo(fn, newfn, pool, args)
    #if False:
    #    converter.correctAllImages(pool, args)
    #converter.correctAllVideos()
