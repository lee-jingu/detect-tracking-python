import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--model', type=str, default='model.pth', help='model file')
    parser.add_argument('--image', type=str, default='test.jpg', help='image file')
    parser.add_argument('--output', type=str, default='output.jpg', help='output file')
    return parser.parse_args()

def main():
    args = get_args()
    pass

if __name__ == '__main__':
    main()