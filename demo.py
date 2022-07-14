import argparse

from inference.opencv import VideoReader, ImageReader, WEBCAM
from inference.opencv import Writer
from inference.dataset import DavidDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=WEBCAM, help='source of video or image dir')
    parser.add_argument('--json-format', type=str, default='david', help='source of json file')
    parser.add_argument('--output-dir', type=str, default='outputs', help='output file')
    parser.add_argument('--padding-size', type=int, nargs=2, default=(1920, 1080), help='padding size')

    return parser.parse_args()

def main():
    args = parse_args()
    args.source = '/home/vv-team/vv-dataset/office/justco_cafe'
    reader = ImageReader(args.source)
    dataset = DavidDataset(path=args.source, padding_size=args.padding_size) if args.json_format == 'david' else None
    writer = Writer(reader=reader, output_dir=args.output_dir)

    for frame, data in zip(reader, dataset):
        for person in data:
            writer.draw_bbox(frame, person.full.xyxy, str(person.id), person.id)
            writer.draw_key_points(frame, person.key_point)
        reader.show(frame)

if __name__ == '__main__':
    main()