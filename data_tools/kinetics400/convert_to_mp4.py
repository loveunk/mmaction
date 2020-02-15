import os, glob, pwd, mmcv
import subprocess


def convert_dir(video_root):
    '''
    Convert all non-mp4 videos to *.mp4
    '''
    videos = set(glob.glob(video_root + "/*")) - set(glob.glob(video_root + "/*.mp4"))

    print('Working on path: {}'.format(video_root))
    print('Videos to be converted: {}'.format(len(videos)))

    for i, video in enumerate(mmcv.track_iter_progress(videos)):
        src = video
        dirname  = os.path.dirname(src)
        basename = os.path.basename(src)
        dst = os.path.join(dirname, basename[0:11] + '.mp4')
        cmd = 'sudo ffmpeg -y -i "{}" "{}" >> {}/ffmpeg.log 2>>&1'.format(src, dst, os.getcwd())

        subprocess.call(cmd, shell=True)


def main():
    # only videos_val contains non-mp4 videos
    relative_dirs = ['../../data/kinetics400/videos_val/*/']
    for relative_dir in relative_dirs:
        video_root = os.path.join(os.getcwd(), relative_dir)
        convert_dir(video_root)


if __name__ == "__main__":
    main()