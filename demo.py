from inference.opencv import VideoReader

# reader = VideoReader(path, width=, height=)
reader = VideoReader(0)

# detector = Detector()
# writer = Writer()

for frame in reader:
    reader.show(frame)
