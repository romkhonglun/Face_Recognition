## Reimplementation ArcFace and SCRFD
- [ArcFace](https://www.kaggle.com/code/nguynhucng/arcface)
- [SCRFD]()
Project folder structure:

```
├── assets/
│   ├── demo.mp4
│   └── in_video.mp4
├── faces/
│   ├── face1.jpg
│   ├── face2.jpg
│   └── ...
├── models/
│   ├── __init__.py
│   ├── scrfd.py
│   └── arcface.py
├── weights/
│   ├── det_2.5g.onnx
│   ├── det_500m.onnx
│   ├── arcface.onnx
├── utils/
│   └── helpers.py
├── main.py
├── README.md
└── requirements.txt
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/romkhonglun/Face_Recognition.git
cd Face_Recognition
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```
3. Put target faces into `faces` folder

```
faces/
    ├── name1.jpg
    ├── name2.jpg
```

Those file names will be displayed while real-time inference.
4. Download pre-trained weights from [here](https://drive.google.com/drive/folders/1v5r_wdYr1S-lStQelnaKEQb_MYKdNUSz?usp=drive_link)
## Usage

run main.py

## Reference

1. https://github.com/deepinsight/insightface/tree/master/detection/scrfd
2. https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
