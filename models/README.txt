YOLOX ONNX models go here.

They are not bundled because they are ~36 MB. Fetch one with:

    python get_models.py

Then run the no-PyTorch backend:

    python app.py --detector yolox --imgsz 640 --conf 0.15 --frame-skip 2
