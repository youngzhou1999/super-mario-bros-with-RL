import moviepy.editor as mpy
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--world', type=int, default=1)
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--iter', type=str, default='')
args = parser.parse_args()

content = mpy.VideoFileClip("output/video_{}_{}_{}.mp4".format(args.world, args.stage, args.iter))

gif = content.subclip()

gif.write_gif("demo/demo_{}_{}.gif".format(args.world, args.stage))