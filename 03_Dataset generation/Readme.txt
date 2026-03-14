Dependencies:

pip install ultralytics



Kør sådan:

python auto_label_to_mot.py --video input.mp4 --model yolov8n.pt --output dataset_out --calibration calibration/min_fil.json --draw-preview

Hvis du vil gemme hver frame i stedet for hver 3. frame:

python auto_label_to_mot.py --video input.mp4 --model yolov8n.pt --output dataset_out --save-every-nth-frame 1

Hvis du vil beholde personer udenfor banen også:

python auto_label_to_mot.py --video input.mp4 --model yolov8n.pt --output dataset_out --calibratio