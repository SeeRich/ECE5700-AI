# ECE5700-AI
Purdue ECE5700 - Artifical Intelligence


### For converting jupyter notebooks to PDFs:
```shell
# For webpdf support
playwright install chromium
# For generic pdf support
brew install --cask mactex
```

### PROJECT NOTES:
* Take the point_coords from the original SAM model and feed it to the Mobile SAM model SamPredictor then calculate the IoU
* Should probably only output one mask for each point_coord...
* NOTE: Maybe the authors fed in BGR and RGB images to the model?
* Bug where the image was not being sampled at the same points as the original model