# v4 bbox ensemble

## Models
- frcnn_x101
- retinanet_r101
- frcnn_r101

## Train
```powershell
D:\conda-envs\cv-engineering\python.exe train_frcnn_x101.py --gpu 0
D:\conda-envs\cv-engineering\python.exe train_retinanet_r101.py --gpu 1
D:\conda-envs\cv-engineering\python.exe train_frcnn_r101.py --gpu 0
```

## Outputs
`detectron2_training/output_v4_ensemble/<model_name>`

## Ensemble inference
```powershell
D:\conda-envs\cv-engineering\python.exe ensemble_infer.py
```
