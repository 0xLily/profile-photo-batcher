# Profile Photo Batcher

Batch process profile photos into consistently cropped square headshots.

This tool is designed for local use from the command line and works well for:

- team directories
- lab or company websites
- student rosters
- event badges
- profile-photo cleanup before upload

## What it does

- Process a whole folder of profile photos in one run
- Detect the main face in each image with OpenCV
- Recenter the crop so the head sits near the middle
- Crop to the requested aspect ratio
- Resize to an exact target size
- Accept size input in either pixels or inches
- Write a `processing_report.txt`
- Build a zip archive of the processed outputs

## Requirements

- Python 3.14
- macOS
- packages listed in `requirements.txt`

## Install

From the project folder:

```bash
python3 -m pip install -r requirements.txt
```

## Run Interactively

```bash
python3 app.py
```

The script will prompt for:

- Input folder
- Output folder
- Width
- Height
- Units (`pixels` or `inches`)
- DPI
- Face fill

## Run With Arguments

Website-friendly square headshots:

```bash
python3 app.py \
  --input-dir "/path/to/input" \
  --output-dir "/path/to/output" \
  --width 400 \
  --height 400 \
  --units pixels \
  --dpi 300 \
  --face-fill 0.50
```

Passport-style or print sizing in inches:

```bash
python3 app.py \
  --input-dir "/path/to/input" \
  --output-dir "/path/to/output" \
  --width 2 \
  --height 2 \
  --units inches \
  --dpi 300 \
  --face-fill 0.50
```

## Output

The output folder will contain:

- processed image files
- `processing_report.txt`
- `profile_photos_<width>x<height>.zip`

## Recommended Settings

- `400 x 400` pixels: safest for websites and least likely to need aggressive cropping
- `450 x 450` pixels: slightly larger web profile image
- `600 x 600` pixels: larger output, but more demanding on tightly framed originals
- `face-fill 0.50`: good default when you want more consistent head size
- `face-fill 0.38` to `0.45`: safer if source images are tight and you want more surrounding hair/headroom

## Limitations

- This is still a batch cropper, not a full studio retouching tool.
- Very unusual angles, heavy occlusion, or tiny faces may still need manual review.
- Different original camera distances can still create some variation, though the tool now tries to normalize head size.

## Share On GitHub

The easiest way to share this tool is to put this folder in its own GitHub repository.

From `profile_photo_batcher`:

```bash
git init
git add .
git commit -m "Initial profile photo batcher"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/profile-photo-batcher.git
git push -u origin main
```

Then other people can use:

```bash
git clone https://github.com/YOUR-USERNAME/profile-photo-batcher.git
cd profile-photo-batcher
python3 -m pip install -r requirements.txt
python3 app.py
```
