# models/

Place your trained **YOLO26** weights here:

```
models/
└── best.pt   ← copy here after training
```

The model is trained in:
`notebooks/GARBAGE_CLASSIFICATION_3_Computer_Vision_Model_yolo26.ipynb`

After training, `best.pt` is saved to `runs/detect/train/weights/best.pt`.
Copy it here:

```bash
copy runs\detect\train\weights\best.pt models\best.pt
```

## Git LFS

Model weights are excluded from regular git tracking (see `.gitignore`).
To share weights via Git LFS:

```bash
git lfs track "models/*.pt"
git add .gitattributes models/best.pt
git commit -m "Add trained YOLO26 weights"
```

