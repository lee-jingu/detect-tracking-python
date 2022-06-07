import os

def get_parse():
    pass

def setup_requirements():
    pass

def setup_docker():
    pass

def build_docker():
    pass

def main():
    args = get_parse()

    if args.setup_docker:
        setup_docker()
    
    if args.build_docker:
        build_docker()
    
    if args.setup_requirements:
        setup_requirements()


if __name__ == '__main__':
    main()