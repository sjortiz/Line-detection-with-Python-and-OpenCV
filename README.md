# Line detection with Python and OpenCV

## Project description

**[TLDR;]**

Implement computer vision to detect the street lines in an image or video.

**[Long-story-short]**

This project uses OpenCV to parse the image into arrays to applya set of transformation over itself such as: `COLOR_RGB2GRAY`, `GaussianBlur` and `Canny`

It also makes use of numpy to generate a secondary images to graph a section of interest, and apply bit-wise operations(&) to extrapolate only the relevant pixels.

Futher more, I use it to calculate the intercepts and slopes, then to extract the means of the combinations of this lines to get a seudo-best-fit-line (something alike the line of best fit)

Thus making the use of openvc again to draw the lines over the original image, using the lines resulting from the numpy part of the process, a straight forward task.

**[Bonus]**

I implemented the same process without changes to do line detection on videos,
just by applying the process over each frame while it's being readed, this serves as a good example on how perform this task over a streaming service.

## Installation
`pip install -r requirements.txt`

## Execution
`python lanes.py`
