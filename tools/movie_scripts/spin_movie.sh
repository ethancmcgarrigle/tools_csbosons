#!/bin/bash
#filename = 'density_n_00'

ffmpeg -r 15 -f image2 -s 1920x1080 -i spin_n_%02d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
