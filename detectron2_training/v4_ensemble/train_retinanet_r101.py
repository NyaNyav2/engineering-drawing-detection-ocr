from train_ensemble import main

if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0], '--model', 'retinanet_r101'] + sys.argv[1:]
    main()
