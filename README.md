# alfs_py
A Python-based Viewer for Airborne Light-Field Sampling Data implemented via ModernGL

## No OpenCV autocomplete in PyCharm

- In preferences, select Python Interpreter
- In the interpreter drop down select show all
- A list of all configured interpreters is show with the current interpreter already hi-lighted
- With the interpreter still highlighted, click the icon that shows a folder and subfolder at the top. Tool tip should say "Show Interpreter Paths".
- Click the + button and add the following path: ```<full path to the venv>/Lib/site-packages/cv2```

## Docker

There are two docker images that can be used for orthografic and ALFS-based rendering of a dataset created with the BAMBI pipeline.

### Orthografic rendering

```cli
docker build --tag orthorender -f Dockerfile_Orthografic .
```

CPU only:
```cli
docker run --rm -v C:\Users\P41743\Desktop\test_with_correction_info:/input -v C:\Users\P41743\Desktop\test_projection:/output --name orthorenderer -e INPUT_DIR="/input" -e OUTPUT_DIR="/output" orthorender
```

GPU accelerated:
```cli
docker run --rm -v C:\Users\P41743\Desktop\test_with_correction_info:/input -v C:\Users\P41743\Desktop\test_projection:/output --ipc=host --gpus '"device=0"' --name orthorenderer -e INPUT_DIR="/input" -e OUTPUT_DIR="/output" orthorender
```

Required environment variables (based on your mounted volumes):
- INPUT_DIR: Path to the input folder
- OUTPUT_DIR: Path to the output folder

Additional (optional) environment variables:
- SPLITS: Splits that should be orthografic projected. Comma separated list as string (default: "train,val,test")
- CAMERA_DISTANCE: Camera distance (default: 10.0)
- ORTHO_WIDTH: Ortho width in units of DEM e.g. meter (default: 70)
- ORTHO_HEIGHT: Ortho height in units of DEM e.g. meter (default: 70)
- INPUT_WIDTH: Width of input images in pixel (default: 1024)
- INPUT_HEIGHT: Width of input images in pixel (default: 1024)
- RENDER_WIDTH: Render width in pixel (default: 2048)
- RENDER_HEIGHT:  Render height in pixel (default: 2048)
- INITIAL_SKIP:  The number of shots read from the Poses JSON file to be skipped from the beginning (default: 0)
- ADD_BACKGROUND:  Whether to overlay the result over a render of the background object (default: 1; True)
- FOVY:  The FOV along the Y-axis when using a perspective camera (default: 50.0)
- ASPECT_RATIO:  The aspect ratio when using a perspective camera (default: 1.0)
- SAVE_LABELED_IMAGES: Additional to raw projected images also images with drawn labels are saved (default: 0; False)

### ALFS based rendering

TODO
