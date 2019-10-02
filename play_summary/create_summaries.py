import os
import glob
import sys
sys.path.append('/home/emma/summary_evaluation/')
from play_summary.videosum_dataset import VideosumDataset
import datetime
import shutil
from subprocess import Popen, PIPE



def create_summaries(videosumdataset, start_clean=False):

    videos = glob.glob(os.path.join(videosumdataset.ds_loc, 'videos')+'/*')
    summaries_dir = os.path.join(videosumdataset.ds_loc, 'summaries')

    try:
        os.mkdir(summaries_dir)
    except:
        if start_clean:
            shutil.rmtree(summaries_dir)
            os.mkdir(summaries_dir)

    for i, (movie_name, segments) in enumerate(sorted(videosumdataset.segments.items())):

        # vid_nr = int(movie_name.split('_')[0])
        # if vid_nr > 54:
        #     continue

        mn = movie_name.replace(' ', '_')
        input_video = [x for x in videos if os.path.basename(x).startswith(mn)][0]

        #input_video = videos[i]
        extension = input_video.split('.')[-1]
        segments_dir = os.path.join(videosumdataset.ds_loc, 'segments', mn)

        try:
            os.mkdir(segments_dir)
        except:
            print('{} dir exists'.format(movie_name))
            if start_clean:
                shutil.rmtree(segments_dir)
                os.mkdir(segments_dir)

        file_name = '{}/{}_list.txt'.format(videosumdataset.ds_name, mn)

        try:
            os.remove(file_name)
        except:
            pass
        f = open(file_name, "w+")

        print(movie_name)
        for i, seg in enumerate(segments):
            #continue
            start_time, duration, frame_times = seg

            # print(start_time, duration)
            if duration < 2 and videosumdataset.ds_name == 'moviesum':
                continue
            ss, t = convert_time(start_time), convert_time(duration)


            output_video = os.path.join(segments_dir, '{}_{}_{}.{}'.format(i, int(start_time), int(duration), extension))

            #extract_segs_command = 'ffmpeg -i {i} -ss {ss} -t {t} -c:v copy -an -f mp4 {o}'
            #extract_segs_command = 'ffmpeg -i {i} -ss {ss} -t {t} -c:v libx264 -c:a libfdk_aac {o}'
            # extract_segs_command = 'ffmpeg -i {i} -ss {ss} -t {t} -c copy map 0 {o}'

            if videosumdataset.ds_name == 'moviesum':
                #extract_segs_command = 'ffmpeg -nostats -loglevel error -ss {ss} -i {i} -map 0 -c copy -t {t} {o}'
                extract_segs_command = 'ffmpeg -nostats -loglevel error -ss {ss} -i {i} -vcodec copy -acodec copy -t {t} {o}'
            else:
                extract_segs_command = 'ffmpeg -nostats -loglevel error -ss {ss} -i {i} -acodec libmp3lame -vcodec libx264 -t {t} {o}'

            extract_segs_command = extract_segs_command.format(i=input_video, ss=ss, t=t, o=output_video)


            os.system(extract_segs_command)
            f.write('file {}\n'.format(output_video))

        f.close()

        print('--' * 10)
        print('concatenate {}'.format(mn))
        print('--' * 10)
        output_summary = os.path.join(summaries_dir, '{}_trailer.{}'.format(mn, extension))
        concat_segs_command = 'ffmpeg -loglevel error -f concat -safe 0 -i {} -c copy {}'.format(file_name, output_summary)

        os.system(concat_segs_command)
        print('done')

def convert_time(seconds):
    ''' convert seconds to string of min:sec:ms '''
    return '0' + str(datetime.timedelta(seconds=seconds))



if __name__ == '__main__':
    local = False
    sc = True
    moviesum = VideosumDataset(name='MovieSum', num_classes=2, local=local)
    create_summaries(moviesum, start_clean=sc)

    # tvsum = VideosumDataset(name='TVSum', num_classes=1, local=local)
    # create_summaries(tvsum, start_clean=sc)
    # #
    summe = VideosumDataset(name='SumMe', num_classes=1, local=local)
    create_summaries(summe, start_clean=sc)
