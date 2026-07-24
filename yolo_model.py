from ultralytics import YOLO

def main():
    # Load YOLOv8 pretrained
    model = YOLO("yolov8.pt")

    # https://docs.ultralytics.com/modes/train/#musgd-optimizer
    print("Starting training...")
    model.train(
            data="dataset.yaml",         #path to dataset configuration file
            epochs = 100,                #number of complete training passes
            batch = 4,                   #feed in batches of 4 images at a time
            patience = 5,                #halts training if no improvement after this many epochs
            imgsz = 640,                 #input image resolution
            project = "Program",         #root directory for saved training run outputs
            name = "norfisk_run",        #subfolder for specific training run
            overwrite = True,            #overwrite directory instead of making a new one
            rect = True,                 #helps shorten processing time by removing useless pixels
            cos_lr = False,              #no cosine learning rate (default, uses linear decay)
            close_mosaic = 0,            #disable data augmentation
            lr0 = 0.01,                  #inital learning rate (default)
            lrf = 0.01,                  #final learning rate as a fraction of lr0 (final_lr = lr0*lrf)
            val = True,                  #evaluates model performance on the validation set during training
            plots = True,                #generates and saves training/validation loss charts and metric plots
            hsv_h = 0.015,               #randomly adjust image colour hue by +/- 1.5%
            hsv_s = 0.7,                 #randomly adjust image colour saturation by +/- 70%
            hsv_v = 0.4,                 #randomly adjust image brightness by +/- 40% (these help train the model for bad water conditions)
            translate = 0,               #disable image translation augmentation
            scale = 0,                   #disable image scaling/zooming augmentation
            fliplr = 0,                  #disable image translation augmentation
            mosaic = 1.0,                #collates 4 images of fish into one to train with multiple fish in a frame
            erasing = 0                  #disable random erasing data augmentation
            )

    best_model = YOLO("Program/norfisk_run/weights/best.pt")

    # https://docs.ultralytics.com/modes/predict/#inference-sources
    results = best_model.predict(
    source="dataset/images/test",        #directory path with unseen test images
    save=True,                           #saves images with boxes
    conf=0.5                             #confidence threshold
    )

if __name__ == "__main__":
    main()
