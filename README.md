# YOLOv8 Person Detector

Fine-tune YOLOv8 on the CrowdHuman dataset, with its large number of labelled people,
to detect people more accurately than the stock COCO model, then export it to run on
the CPU of an Intel NUC. ONNX and OpenVINO were both exported and
timed to decide which runtime to use. OpenVINO won one test and tied the other, and it
is the one this project runs on, quantized to INT8. A second, short fine-tune on frames
from the camera itself taught the model that the family dog is not a person. A second
model, trained the same way in two stages, detects the dog as a class of its own. It
does better on the camera than the one-class model, and it is the one the NUC runs (see
[Two classes: person and dog](#two-classes-person-and-dog)).

The pipeline is split into small, independent stages so each one can be run and
inspected on its own:

```
prepare  ->  train  ->  val  ->  export  ->  bench  ->  predict  ->  deploy
```

## Why CrowdHuman

COCO already knows the `person` class. To raise the detection accuracy on people,
the model is fine-tuned on CrowdHuman: 15k images with a wide variety of people,
poses, sizes, scenes and lighting.
People who overlap heavily or have half their body hidden are not the target here.
This is a security camera, and crowds are not the usual scene.
We keep only the `person` class and drop everything else.

### About the labels

This project trains on the **official** CrowdHuman ground truth: `annotation_train.odgt`
and `annotation_val.odgt`, 15k train images and the separate 4,370-image val split.

Those files are line-delimited JSON, not YOLO `.txt`, so `prepare` converts them. Three
decisions are baked into that conversion:

- **`vbox` is the box.** Each annotation carries `fbox` (full body, occluded parts
  included), `vbox` (visible part only) and `hbox` (head). `vbox` boxes what can
  actually be seen of each person, the same convention as COCO and as the camera frames
  used later, so it is the default; `--box fbox` and `--box hbox` are there as well.
- **`mask` regions are dropped.** 85,809 of them in train. They mark crowds the
  annotators left unlabelled: not people, but full of people. Ultralytics has no
  ignore-region support, so detections landing there count as false positives. This
  caps the precision that can be measured, regardless of how well the model trains.
- **Boxes are clipped to the frame.** A few annotations run slightly past the edge of
  the picture, 4.3% of the train boxes, and are trimmed to it.

The labels are the official CrowdHuman ground truth, converted to YOLO format; no
detector is involved. That gives about 23 person boxes per image on average.

## Project layout

```
.
|-- src/person_detector/
|   |-- odgt.py        # official .odgt -> person-only YOLO labels
|   |-- train.py       # fine-tune from COCO-pretrained weights
|   |-- evaluate.py    # mAP/precision/recall + latency benchmark
|   |-- export.py      # .pt -> ONNX or OpenVINO
|   |-- predict.py     # run the model on your own images or video
|   |-- cli.py         # `person-detect` command
|   `-- utils.py       # device / seeding / weight lookup
|-- deploy/            # Docker image for the NUC (detect.py, Dockerfile, compose)
|   `-- models/best_openvino_model/   # deployed model, INT8 (included)
|-- runs/person/twoclass_stage2_camera/weights/last.pt   # final model (included)
|-- scripts/           # end-to-end run, fine-tuned vs stock comparison
|-- docs/comparison/   # figures and images used below
|-- docs/camera/       # frames from the home camera used below
|-- pyproject.toml
`-- requirements.txt
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

`pip install -e .` exposes the `person-detect` command. If you'd rather not install the
package, `pip install -r requirements.txt` and call `python -m person_detector.cli ...`
instead.

Two notes on the optional pieces:

- With `onnxruntime-gpu` already installed, use `--no-deps` or drop the `onnxruntime`
  line, or pip will put the CPU build on top of it. Ultralytics' auto-install would do
  the same, so `person_detector/__init__.py` sets `YOLO_AUTOINSTALL=false` on import
  (export `YOLO_AUTOINSTALL=true` to turn it back on). If it already happened,
  uninstall both packages and reinstall `onnxruntime-gpu`.
- The INT8 export (`--int8`) needs NNCF, which `pip install -e .` leaves out. It is an
  optional extra called `quantize`, so install the package with it:

  ```bash
  pip install -e ".[quantize]"   # the package plus NNCF; keep the quotes
  ```

  The quotes stop shells such as zsh from reading the brackets as a file pattern. With
  the package already installed, the same command just adds NNCF. `requirements.txt`
  already lists NNCF, so the `pip install -r requirements.txt` route needs nothing more.

## Usage

**Weights included.** The repository holds only two sets of weights: the final
two-class model (`runs/person/twoclass_stage2_camera/weights/last.pt`) and its INT8
export in `deploy/models/best_openvino_model/`, ready to deploy (step 7). The other
models this README refers to (`oneclass_stage1_crowdhuman`, `oneclass_stage2_camera`,
`twoclass_stage1_crowdhuman_coco`) are not included: the commands that use them need
those models trained first, following the steps below.

### 1. Prepare the dataset

Lay the download out like this:

```
~/datasets/crowdhuman/
|-- images/train/            15,000 jpg   (CrowdHuman_train01..03 merged)
|-- images/val/               4,370 jpg   (CrowdHuman_val)
|-- annotation_train.odgt
`-- annotation_val.odgt
```

Then write the labels in place. Nothing is copied or symlinked; Ultralytics finds
`labels/` by swapping it for `images/` in the path:

```bash
person-detect prepare --data-root ~/datasets/crowdhuman
```

The command does three things:

- Reports, for train and val, how many person boxes it kept, how many `mask` regions it
  dropped and how many boxes it clipped to the edge of the picture.
- Writes `data.yaml` next to the images.
- Deletes any `*.cache` files Ultralytics left behind. They hold the labels already
  read, and when Ultralytics finds one it does not read the `.txt` files again. Left in
  place, the next training run would use the old labels even if you had just
  regenerated them.

### 2. Fine-tune

```bash
person-detect train --data ~/datasets/crowdhuman/data.yaml --model yolov8n.pt \
  --name oneclass_stage1_crowdhuman --epochs 100 --patience 20
```

Training starts from the public `yolov8n.pt` weights. Swap in `yolov8s.pt` for a bit
more accuracy at the cost of speed. The second fine-tune on camera frames is described
in [Home security camera](#home-security-camera). A finished run is reused
automatically; pass `--force-retrain` to start over.

### 3. Validate

```bash
person-detect val --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --data ~/datasets/crowdhuman/data.yaml
```

It runs the model over the `val` split of `data.yaml`, the 4,370 CrowdHuman val images
here, and prints four numbers: mAP50, mAP50-95, precision and recall (explained under
[Results](#results)). It runs on the GPU when there is one, and on the CPU otherwise.
`--imgsz` sets the input size, 640 by default, the same as in training.

The same command scores any other model on any other set: pass its weights with
`--weights` and a `data.yaml` whose `val` points at the images to score.

### 4. Export

```bash
person-detect export --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --format openvino
person-detect export --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --format openvino --half
```

That writes `best_openvino_model/`. OpenVINO is the runtime this project deploys on the
NUC (see [CPU latency](#cpu-latency) for why), in its INT8 version (see [INT8](#int8)).
The ONNX export, which it was measured against, is still available:

```bash
person-detect export --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --format onnx
```

`--half` writes an FP16 IR: half the size, no calibration needed, no meaningful accuracy
loss. The throughput gain is small unless the CPU handles FP16 natively, since the
runtime converts back to FP32 at load.

`--int8` runs post-training quantization of the OpenVINO model. It needs NNCF (only
this path does) and **requires `--data`**, since it has to run real images through the
network to set the INT8 scale of each layer. The other exports just convert the weights
and don't need it:

```bash
person-detect export --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --format openvino \
  --int8 --data ~/datasets/crowdhuman/data.yaml
```

The model deployed on the NUC is calibrated on its own training set instead, camera
frames included (see step 7).

Calibration images have to look like the ones the model will see in use. Ultralytics
takes them from its `data` argument, and falls back to `coco8.yaml` (eight COCO images)
when it is missing, giving a degraded INT8 model with nothing in the output to say so.
The export step raises instead.

INT8 lands in `best_int8_openvino_model/`, a different folder from the FP32/FP16
`best_openvino_model/`, so both can be benchmarked side by side.

### 5. Benchmark

The benchmark accepts any weights Ultralytics can load, so the same command compares
the PyTorch model against its exported variants and the choice of format rests on
numbers:

```bash
person-detect bench --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --images ~/datasets/crowdhuman/images/val --device cpu
person-detect bench --weights runs/person/oneclass_stage1_crowdhuman/weights/best.onnx --images ~/datasets/crowdhuman/images/val --device cpu
person-detect bench --weights runs/person/oneclass_stage1_crowdhuman/weights/best_openvino_model --images ~/datasets/crowdhuman/images/val --device cpu
```

The time covers the whole `predict` call: reading the image, resizing, the forward pass
and NMS (non-maximum suppression).

### 6. Run it on your own images

```bash
person-detect predict --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt --source path/to/images
```

`--source` takes a single image, a folder or a video. The annotated copies go to
`runs/person/predict/` (overwritten on each run, change it with `--name`). `--conf`
sets the confidence cut, 0.25 by default; `--save-txt` also writes the boxes as YOLO
`.txt` files.

`scripts/run_pipeline.sh <crowdhuman_root>` runs steps 1-5 in order.

### 7. Deploy on the NUC with Docker

The NUC runs **Ubuntu Server 24.04 LTS** with Docker Engine, and the detector runs on
its CPU. OpenVINO and ONNX Runtime live inside the container, so the host only needs
Docker.

`deploy/` holds everything the NUC needs. `detect.py` reads the camera, runs the
exported model with OpenVINO or ONNX Runtime and writes what it finds. It does not use
Ultralytics or PyTorch, so the image stays small, but it reproduces the same
preprocessing (grey letterbox) and NMS: on 30 val images it returns exactly the same
519 boxes as Ultralytics with either runtime.

Everything is prepared on the development machine and then copied to the NUC.

**On the development machine**

The repository already ships the INT8 OpenVINO model in `deploy/models/`, so with
OpenVINO the export below can be skipped; it is needed only for ONNX or after a
retrain. Building the Docker image, further down, is needed either way.

Pick one runtime, export the model for it and copy it into `deploy/models/`. The model
is the two-class one, fine-tuned on camera frames (`runs/person/twoclass_stage2_camera`),
the only one kept in this repository. For OpenVINO, the INT8 version:

```bash
person-detect export --weights runs/person/twoclass_stage2_camera/weights/last.pt --format openvino \
  --int8 --data ~/datasets/camera_persons_dogs/calib.yaml
rm -rf deploy/models/best_openvino_model   # replaces the copy shipped with the repo
cp -r runs/person/twoclass_stage2_camera/weights/last_int8_openvino_model deploy/models/best_openvino_model
```

`calib.yaml` is the dataset's `data.yaml` with its val split pointed at the training
list (`train_mix.txt`: the 517 camera frames and the 2,500 general images), so the
calibration sees the same mix the model was trained on and never the test frames.
Calibrated on the camera frames alone, the INT8 model kept a few person boxes on the
dog; with the whole mix it gives none at `CONF` 0.5 (see
[Compared with the one-class model](#compared-with-the-one-class-model)).

For ONNX:

```bash
person-detect export --weights runs/person/twoclass_stage2_camera/weights/last.pt --format onnx
mkdir -p deploy/models
cp runs/person/twoclass_stage2_camera/weights/last.onnx deploy/models/best.onnx
```

Build the image and export it to a file. Run from `deploy/`, `docker save` writes
`person-detector.tar` there:

```bash
cd deploy                                                  # docker-compose.yml and Dockerfile are here
docker compose build                                       # build the image from the Dockerfile
docker save person-detector:latest -o person-detector.tar  # export the image to a single file
```

Docker itself is copied too, as the `.deb` packages for Ubuntu 24.04 from
Docker's download site (`linux/ubuntu/dists/noble/pool/stable/amd64/`): `containerd.io`,
`docker-ce`, `docker-ce-cli`, `docker-buildx-plugin` and `docker-compose-plugin`. Put
them in a `docker-debs/` folder. Despite the name, `amd64` is just 64-bit x86 and covers
Intel CPUs as well, so it is the right folder for the NUC.

Copy the following to the home folder of the NUC, renaming `deploy/` to
`person-detector/`:

```
~/
|-- person-detector/
|   |-- Dockerfile
|   |-- docker-compose.yml
|   |-- detect.py
|   |-- requirements.txt
|   |-- .dockerignore
|   |-- person-detector.tar          # the saved image
|   |-- events/                      # empty; snapshots are written here
|   `-- models/
|       |-- best_openvino_model/     # OpenVINO: the .xml, .bin and metadata.yaml
|       `-- best.onnx                # or ONNX, only the one you exported
`-- docker-debs/
    |-- containerd.io_*.deb
    |-- docker-ce_*.deb
    |-- docker-ce-cli_*.deb
    |-- docker-buildx-plugin_*.deb
    `-- docker-compose-plugin_*.deb
```

The `*` in the `.deb` names stands for the version number. `events/` has to be copied
too. The container runs as user 1000, group 1000 (`user: "1000:1000"` in
`docker-compose.yml`), which is the first account Ubuntu creates, so the snapshots
belong to you and not to root. Check yours on the NUC with `id -u` and `id -g`; if they
print, say, `1001` and `1001`, change that line to `user: "1001:1001"`.

**On the NUC**

Install Docker, once. The packages need `iptables` and `nftables`, the Linux firewall
tools Docker uses to give the container its own network and a way out to the camera.
Ubuntu Server 24.04 ships both by default:

```bash
sudo dpkg -i ~/docker-debs/*.deb  # install the local .deb files, nothing downloaded
sudo usermod -aG docker $USER      # add your user to the docker group, to run docker
                                   # without sudo; then log out and back in
```

Load the image, set `RUNTIME` in `docker-compose.yml` to `openvino` for the INT8 model
shipped with the repository (or `onnx` if you exported ONNX), set the camera in
`SOURCE`, and start it:

```bash
cd ~/person-detector                # the deploy/ folder copied to the NUC
docker load -i person-detector.tar  # load the image from the .tar into Docker
docker compose up -d                # start the container in the background
docker compose logs -f              # follow its output live; Ctrl+C stops watching only
```

Settings live in `docker-compose.yml`:

| Variable     | Default  | What it does                                          |
|--------------|----------|-------------------------------------------------------|
| `RUNTIME`    | required | `openvino` or `onnx`; both are installed in the image |
| `SOURCE`     | required | RTSP URL, video file, image folder or USB index (`0`) |
| `CONF`       | `0.5`    | minimum confidence for a detection (person or dog)    |
| `MIN_HEIGHT` | `0.05`   | smallest person box kept, as a fraction of the frame height |
| `FPS`        | `5`      | frames per second taken from the camera               |
| `COOLDOWN`   | `10`     | minimum seconds between two saved snapshots           |

`SOURCE` decides what is read, from the value alone:

| `SOURCE`                                          | Read as                   | Runs               |
|---------------------------------------------------|---------------------------|--------------------|
| `"rtsp://user:password@192.168.1.50:554/<path>"`  | IP camera                 | until stopped      |
| `"0"`                                             | USB camera                | until stopped      |
| `/data`                                           | every image in the folder | until the last one |
| `/data/photo.jpg`                                 | a single image            | once               |
| `/data/clip.mp4`                                  | a video                   | until it ends      |

A path that does not exist stops at once with `source not found`.

**IP camera.** The path after the port (`/stream` in the example) depends on the camera
brand. Write the address in quotes, so YAML always reads it as text: without them, a
`#` in the password, for instance, would turn the rest of the line into a comment.

```yaml
SOURCE: "rtsp://admin:secret@192.168.1.50:554/stream"
```

The user and password go inside the URL, and the `@` marks where the password ends and
the IP begins. If the password contains characters that mean something in a URL, they
have to be written encoded, or the address is read wrong. For example, the password
`Home@2024` is written `Home%402024`:

```yaml
SOURCE: "rtsp://admin:Home%402024@192.168.1.50:554/stream"
```

| Character in the password | Written in the URL |
|---------------------------|--------------------|
| `@`                       | `%40`              |
| `:`                       | `%3A`              |
| `/`                       | `%2F`              |
| `#`                       | `%23`              |
| `?`                       | `%3F`              |
| `%`                       | `%25`              |
| space                     | `%20`              |

The password on the camera stays the same; it is only written this way inside the URL.
A password of only letters and digits needs no encoding.

**USB camera.** Set `SOURCE: "0"` and also uncomment the two `devices:` lines in
`docker-compose.yml`.

To run on images or a video from disk, mount their folder and point `SOURCE` at it.
`docker compose run` goes through them once, without the automatic restart that would
otherwise start over when it finishes:

```bash
docker compose run --rm -e RUNTIME=<openvino|onnx> -e SOURCE=/data -e COOLDOWN=0 \
  -v ./data:/data:ro detector
```

`./data` is the folder on the NUC, next to `docker-compose.yml`; `/data` is the same
folder as the container sees it, which is what `SOURCE` points to, and `ro` keeps it
read-only. With a folder, it reads the `.jpg`, `.jpeg`, `.png`, `.bmp` and `.webp`
files in it by name, not in subfolders, and stops after the last one. `detector` is
the service name in `docker-compose.yml`.

The detector finds both classes, people and the dog, but only people raise an alert.
Every frame with people in it becomes one JSON line in the container log (number of
people, their boxes, the dog's boxes if it is in the frame, latency), and a snapshot
with the boxes drawn goes to `events/`, people in red and the dog in blue. A frame with
only the dog writes nothing. It always works on the
newest camera frame, so it never falls behind, and it reconnects on its own if the
stream drops. `docker-compose.yml` also sets `restart: unless-stopped`, so once started
with `docker compose up -d` it comes back after a crash or a reboot of the NUC, until
you stop it yourself with `docker compose stop` or `docker compose down`.

The same script runs without Docker, which is handy to try it on a recording first. Use
its own venv: `deploy/requirements.txt` pulls the CPU `onnxruntime`, which would clash
with an `onnxruntime-gpu` in the training environment.

```bash
python -m venv .venv-deploy && source .venv-deploy/bin/activate
pip install -r deploy/requirements.txt
python deploy/detect.py --runtime <openvino|onnx> --model <exported model> \
  --source path/to/video.mp4 --save-dir events
```

## Results

The fine-tuned model is yolov8n at 640 px, batch 16, 100 epochs. Both models are scored
on the 4,370 CrowdHuman val images with `vbox` labels:

| Model                  | mAP50 | mAP50-95 | Precision | Recall |
|------------------------|-------|----------|-----------|--------|
| yolov8n COCO (stock)   | 0.599 | 0.355    | 0.725     | 0.511  |
| yolov8n fine-tuned     | 0.771 | 0.494    | 0.824     | 0.672  |

Precision and recall are taken at the confidence threshold that best balances the two
for each model, as Ultralytics reports them: about 0.33 for the fine-tuned one and
0.13 for the stock one. The mAP scores don't depend on a threshold.

Both mAP scores go from 0 to 1, higher is better. They reward finding every person with
few false alarms. **mAP50** counts a box as right when it overlaps the real person by at
least 50%. **mAP50-95** repeats the same calculation ten times, demanding more overlap
each time (50%, 55%, 60% … 95%), and averages the ten results. A box only counts at the
top thresholds if it sits almost exactly on the person, so this score also measures how
tightly the boxes fit. In both, the percentages refer to box overlap, not to the
model's confidence.

The stock row is the public `yolov8n.pt` with only its `person` class, before any
fine-tuning; see [Was the fine-tune worth it?](#was-the-fine-tune-worth-it).

The COCO weights already know what a person is, so epoch 1 starts at 0.671 (mAP50) and
0.391 (mAP50-95). By epoch 50 those are 0.764 (mAP50) and 0.487 (mAP50-95), and the
second half of the run only adds 0.007 to each. The val losses keep going down to the
end, so there is no sign of overfitting, but more epochs would not buy much either.

The step down in the train loss at epoch 91 is not a bug: Ultralytics turns mosaic
augmentation off for the last 10 epochs (`close_mosaic=10`), and the plain images are
easier to fit.

### CPU latency

The detector will run on an Intel NUC, so it has to be fast on an Intel CPU. These
timings were taken for one reason only: to choose between ONNX Runtime and OpenVINO.
Everything below is the forward pass alone on an i7-12700H, with a 640x640 input and
the calls back to back.

First with the same input repeated 100 times:

| Runtime        | Size  | Median latency (ms) | FPS |
|----------------|-------|---------------------|-----|
| OpenVINO FP32  | 12 MB | **18.3**            | 55  |
| ONNX Runtime   | 12 MB | 21.3                | 47  |

Then with 100 real val images letterboxed to 640x640, five rounds with the formats
interleaved, taking the median of the five. PyTorch is there as the baseline:

| Format         | Size  | Median latency (ms) | FPS |
|----------------|-------|---------------------|-----|
| PyTorch `.pt`  | 6 MB  | 35.0                | 29  |
| ONNX Runtime   | 12 MB | 25.5                | 39  |
| OpenVINO FP32  | 12 MB | 25.6                | 39  |
| OpenVINO FP16  | 6 MB  | 25.9                | 39  |

With real images the two runtimes are level: the spread between rounds (21 to 30 ms)
is bigger than the gap between them. Both are well ahead of plain PyTorch. OpenVINO
keeps the job anyway: it won the first test, and it is Intel's own runtime, made for
the kind of CPU the NUC has. FP16 runs at the same speed as FP32 and only makes the
file half as big: the CPU computes in FP32, so OpenVINO converts the weights back to
FP32 when it loads the model.

### INT8

Measured on the two models fine-tuned with frames from the home camera: the one-class
one (see [Home security camera](#home-security-camera)) and the two-class one the NUC
runs (see [Two classes: person and dog](#two-classes-person-and-dog)). Both are scored
on the same frames: the 644 held-out camera test frames (see
[Stage 2](#stage-2-camera-frames)) and the CrowdHuman val set. Each INT8 model was
calibrated on images from its own training data, never on a test frame: the one-class
one on its 521 camera frames, the two-class one on its whole training mix, 517 camera
frames and 2,500 general images.

| Model       | OpenVINO | Size   | Camera person mAP50 | Camera person mAP50-95 | Camera dog mAP50 | Camera dog mAP50-95 | CrowdHuman mAP50 | CrowdHuman mAP50-95 |
|-------------|----------|--------|---------------------|------------------------|------------------|---------------------|------------------|---------------------|
| One class   | FP32     | 12 MB  | 0.985               | 0.838                  | -                | -                   | 0.762            | 0.485               |
| One class   | INT8     | 3.5 MB | 0.981               | 0.832                  | -                | -                   | 0.758            | 0.475               |
| Two classes | FP32     | 12 MB  | 0.989               | 0.848                  | 0.872            | 0.634               | 0.758            | 0.480               |
| Two classes | INT8     | 3.5 MB | 0.982               | 0.821                  | 0.867            | 0.621               | 0.754            | 0.470               |

mAP sums up every threshold, but the NUC runs at one. How each INT8 model does at the
`CONF` it runs at is in [Compared with the one-class model](#compared-with-the-one-class-model).

Same timing method as above, five rounds with the three models interleaved:

| OpenVINO          | Same input repeated (ms) | Real camera frames (ms) | FPS  |
|-------------------|--------------------------|-------------------------|------|
| One class, INT8   | 9.4                      | 9.5                     | 105  |
| Two classes, FP32 | 23.6                     | 24.1                    | 41   |
| Two classes, INT8 | **9.4**                  | **9.4**                 | 106  |

INT8 is about two and a half times faster and a quarter of the size. It costs under a
point of mAP50, and up to three points of mAP50-95, on the camera frames and on
CrowdHuman alike. For a camera sampled at a few frames per second that trade is well
worth it, and it leaves the NUC's CPU free for decoding the stream. The dog class adds
nothing to the time: both models run at the same speed. **The NUC runs the two-class
model in INT8.**

### Was the fine-tune worth it?

The obvious question, since `yolov8n.pt` already detects people out of the box, and
the answer decides which weights are worth deploying. Both rows of the Results table
come from the same val pass: same 4,370 images, same labels, same 640 px input.

![Metrics, fine-tuned vs stock](docs/comparison/metrics.png)

![Precision-recall curves](docs/comparison/pr_curve.png)

The Results table gives recall at a single confidence threshold; these figures come
from the whole curve:

| Model                  | Recall (Results table) | Highest recall | Recall with precision > 0.9 |
|------------------------|------------------------|----------------|-----------------------------|
| yolov8n COCO (stock)   | 0.511                  | 0.791          | 0.379                       |
| yolov8n fine-tuned     | 0.672                  | 0.898          | 0.602                       |

The recall is where it shows, the share of the real people the model finds:

- **Highest recall.** However low the confidence goes, the stock model finds no more
  than 79% of the people: some never get a box from it at all. The fine-tuned one gets
  to 90%.
- **Recall with precision > 0.9.** Asking for more than 9 out of 10 boxes to be real
  people, the stock model finds only 38% of the people, and the fine-tuned one 60%.

Three val images, fine-tuned on the left, stock on the right, conf 0.25:

**A park path** (5 people annotated). The fine-tuned model finds all five; the stock
one finds three, and gives each of them a lower score (0.88, 0.86, 0.85 against 0.92,
0.91, 0.90).

![Park path](docs/comparison/example_1.jpg)

**Inside a helicopter** (3 annotated). Same three people, higher scores on each:
0.89, 0.81 and 0.51 against 0.81, 0.65 and 0.33.

![Inside a helicopter](docs/comparison/example_2.jpg)

**Children in a forest** (3 annotated). Higher on all three again, and the child at
the back goes from 0.54 to 0.72.

![Children in a forest](docs/comparison/example_3.jpg)

`scripts/compare_models.py` keeps that check repeatable, for instance after a retrain.
It regenerates all of this (runs both val passes, then draws the images you pass with
`--images`):

```bash
python scripts/compare_models.py --data ~/datasets/crowdhuman/data.yaml \
  --images ~/datasets/crowdhuman/images/val/<name>.jpg ...
```

The same side-by-side can be made for any folder of your own images, to see how
detection holds up on your own scenes before settling on a model:

```bash
person-detect predict --weights runs/person/oneclass_stage1_crowdhuman/weights/best.pt \
  --source path/to/images --compare-base
```

It runs on the development machine, not on the NUC, since it needs Ultralytics and
PyTorch, which the Docker image leaves out. The stock model is `yolov8n.pt` from the
folder you run it in; `--compare-base other.pt` compares against a different one.

## Home security camera

CrowdHuman is mostly eye-level photos of crowds in daylight. A home camera is a
different setting: mounted high and looking down, infrared at night, rarely more than
one or two people in frame, and in this case a small black dog running around the
patio. At first the model often took the dog for a person. The fine-tune described below
cut that down, and the [two-class model](#two-classes-person-and-dog) takes the dog out
of the alerts altogether.

### A second fine-tune with camera frames

To make sure the detector tells people from the family dog, the CrowdHuman model went
through a second, short fine-tune on frames from the camera itself. Its one purpose is
to keep the dog from setting off false detections, without losing anything on people.

The process, step by step:

1. **Collect frames.** Short clips from the camera, by day, at dusk and at night
   (infrared), sampled at one frame per second.
2. **Sort them into two groups.** Frames with the dog alone, and frames with people
   and the dog around.
3. **Label them.** Only people are labelled. The dog is never boxed, so the dog-only
   frames have no labels at all.
4. **Keep whole videos apart for testing.** Clips by day, at dusk and at night, so no
   near-identical frame ends up on both sides.
5. **Mix in part of CrowdHuman.** 2,500 of its images, so the model keeps what it
   already knew about people.
6. **Train again from the CrowdHuman weights**, with a learning rate ten times lower.

| Frames                       | Labels             | Train | Test |
|------------------------------|--------------------|-------|------|
| CrowdHuman, random subset    | its own            | 2,500 | -    |
| No people: the dog alone     | none (background)  | 147   | 130  |
| People with the dog around   | people only        | 374   | 514  |

The 644 test frames are the same ones the two-class model is scored on further down, so
both models are compared on exactly the same frames. 8 of the 130 frames without people
have nobody at all in them, not even the dog.

![Dog alone, people with the dog, night](docs/camera/training_frames.jpg)

The dog is never labelled. CrowdHuman is a dataset of people, not animals, and the
model keeps a single class, `person`. Frames with only the dog go in with no boxes, as
background, which is how YOLO learns that something is *not* a person. Frames with
people and the dog next to them carry boxes on the people only, so the model also sees
the two together and learns to pick out just the people.

```bash
person-detect train --data <camera dataset>/data.yaml \
  --model runs/person/oneclass_stage1_crowdhuman/weights/best.pt --name oneclass_stage2_camera \
  --epochs 30 --patience 30 --optimizer SGD --lr0 0.001
```

### Before and after

On the 644 test frames. The last two columns are counted at conf 0.25, the default of
`predict`: person boxes that land on the dog, and frames without people that get a box.

| Model              | Camera mAP50 | Camera mAP50-95 | Dog as person | Frames without people that alert | CrowdHuman mAP50 | CrowdHuman mAP50-95 |
|--------------------|--------------|-----------------|---------------|----------------------------------|------------------|---------------------|
| CrowdHuman only    | 0.968        | 0.766           | 96            | 38 / 130                         | 0.771            | 0.494               |
| + camera frames    | 0.982        | 0.836           | 25            | 11 / 130                         | 0.764            | 0.486               |

The dog is taken for a person four times less often, the boxes fit people seen from the
camera's angle more tightly (mAP50-95), and the CrowdHuman scores drop by less than a
point, so next to nothing was lost on people in general.

Frames the model never saw in training, before and after (conf 0.25):

![The dog next to a person](docs/camera/before_after_dog.jpg)

![The dog walking beside someone](docs/camera/before_after_dog_2.jpg)

More frames it never saw, as the INT8 model reads them: by day, at dusk and at night,
with one or two people and with the dog around.

![Detections by day, at dusk and at night](docs/camera/detections.jpg)

## Two classes: person and dog

The model above has a single class and learns to ignore the dog. A second model goes a
step further and detects the dog as a class of its own, `dog`, next to `person`. It is
trained in two stages, like the first one: a general dataset, then frames from the
camera.

### Stage 1: CrowdHuman plus COCO dogs

| Source                          | Train images | People  | Dogs  | Val images | People | Dogs |
|---------------------------------|--------------|---------|-------|------------|--------|------|
| CrowdHuman                      | 15,000       | 339,560 | 296   | 4,370      | 99,481 | 86   |
| COCO 2017, images with a dog    | 4,385 (x2)   | 6,402   | 5,500 | 177        | 257    | 218  |

- **People** keep the CrowdHuman `vbox` labels described above, and the COCO labels in
  the dog images.
- **Dogs in CrowdHuman** would otherwise be background, so the stock `yolov8x.pt` looked
  for them. Boxes scored 0.6 or more were kept and checked by eye, statues, toys and
  other animals were taken out, and a few lower-scored dogs confirmed by eye were added.
- **COCO dog images are listed twice** in train, so the dogs carry enough weight next to
  15,000 images of people. That gives 23,770 train images and 4,547 val images.

```bash
person-detect train --data ~/datasets/persons_dogs/data.yaml --model yolov8n.pt \
  --name twoclass_stage1_crowdhuman_coco --epochs 100 --patience 20 --optimizer SGD --lr0 0.01 --batch 8
```

On its val set (`best.pt`, epoch 98):

| Class  | Precision | Recall | mAP50 | mAP50-95 |
|--------|-----------|--------|-------|----------|
| person | 0.888     | 0.606  | 0.764 | 0.488    |
| dog    | 0.726     | 0.743  | 0.751 | 0.578    |
| all    | 0.807     | 0.674  | 0.757 | 0.533    |

On people alone it stays close to the one-class model: 0.764 mAP50 and 0.488 mAP50-95
on the 4,370 CrowdHuman val images, against 0.771 and 0.494.

### Stage 2: camera frames

The same idea as the one-class fine-tune, with the dog labelled this time. Dog boxes in
the camera frames were proposed by `yolov8x.pt` at a low threshold and reviewed one by
one by eye.

| Frames                         | Images | People | Dogs  |
|--------------------------------|--------|--------|-------|
| Train: camera                  | 517    | 526    | 366   |
| Train: stage 1 dataset, subset | 2,500  | 45,606 | 798   |
| Test: camera, held out         | 644    | 631    | 355   |

The 644 test frames come from clips the model never saw, by day, at dusk and at night.

```bash
person-detect train --data ~/datasets/camera_persons_dogs/data.yaml \
  --model runs/person/twoclass_stage1_crowdhuman_coco/weights/best.pt --name twoclass_stage2_camera \
  --epochs 30 --patience 30 --optimizer SGD --lr0 0.001 --batch 8
```

Both camera models, one class and two classes, are used from `last.pt`.

| 644 test frames | Person mAP50 | Person mAP50-95 | Dog mAP50 | Dog mAP50-95 |
|-----------------|--------------|-----------------|-----------|--------------|
| After stage 1   | 0.964        | 0.800           | 0.699     | 0.479        |
| After stage 2   | **0.988**    | **0.848**       | **0.885** | **0.635**    |

On CrowdHuman val, the person class is at 0.759 mAP50 and 0.480 mAP50-95 after stage 2,
against 0.764 and 0.488 after stage 1.

Person mAP on these frames goes from 0.982 for the one-class model to 0.988 (mAP50)
and from 0.836 to 0.848 (mAP50-95), both in PyTorch.

### Compared with the one-class model

Both models as the NUC runs them: INT8, through `deploy/detect.py`, with `MIN_HEIGHT`
0.05, on the same 644 test frames, every label as drawn. A detection counts when it
overlaps a label by IoU 0.5 or more; "dog as person" is a person box that lands on the
dog. The last two columns count frames, which is what sets off an alert: a frame with
people should raise one, and a frame without people should not.

| Model       | CONF | Found | Missed | False alarms | Dog as person | Precision | Recall | F1    | Frames with people that alert | Frames without people that alert |
|-------------|------|-------|--------|--------------|---------------|-----------|--------|-------|-------------------------------|----------------------------------|
| One class   | 0.40 | 601   | 30     | 23           | 13            | 0.963     | 0.952  | 0.958 | 502 / 514                     | 6 / 130                          |
| One class   | 0.50 | 591   | 40     | 13           | 9             | 0.978     | 0.937  | 0.957 | 498 / 514                     | 5 / 130                          |
| One class   | 0.55 | 587   | 44     | 8            | 5             | 0.987     | 0.930  | 0.958 | 495 / 514                     | 2 / 130                          |
| One class   | 0.60 | 584   | 47     | 6            | 4             | 0.990     | 0.926  | 0.957 | 494 / 514                     | 1 / 130                          |
| Two classes | 0.40 | 598   | 33     | 9            | 1             | 0.985     | 0.948  | 0.966 | 501 / 514                     | 1 / 130                          |
| Two classes | 0.45 | 596   | 35     | 6            | 1             | 0.990     | 0.945  | 0.967 | 500 / 514                     | 1 / 130                          |
| Two classes | 0.50 | 590   | 41     | **3**        | **0**         | **0.995** | 0.935  | 0.964 | 499 / 514                     | **0 / 130**                      |
| Two classes | 0.55 | 585   | 46     | 2            | 0             | 0.997     | 0.927  | 0.961 | 498 / 514                     | 0 / 130                          |
| Two classes | 0.60 | 570   | 61     | 2            | 0             | 0.997     | 0.903  | 0.948 | 489 / 514                     | 0 / 130                          |

With a class of its own, the dog stops competing with people. From `CONF` 0.5 the
two-class model never puts a person box on the dog and no frame without people raises
an alert, and it still finds more people than the one-class model at 0.55, which alerts
on two of those frames. The three false alarms left at 0.5 are extra boxes in frames
that do have people, so they raise no alert of their own.

**Whole clips, not single frames.** The camera is a video, so a person only has to be
caught in some of the frames they appear in. Over the 39 clips most test frames come
from (588 frames, 33 clips with people, 458 frames with people):

| Model       | CONF | Clips with people detected | Frames with people without alert | Longest run of such frames | False alert frames | Two false alerts in a row |
|-------------|------|----------------------------|----------------------------------|----------------------------|--------------------|---------------------------|
| One class   | 0.40 | 33 / 33                    | 11 / 458                         | 2                          | 6 (3 clips)        | 2                         |
| One class   | 0.55 | 33 / 33                    | 16 / 458                         | 2                          | 2 (2 clips)        | 0                         |
| Two classes | 0.40 | 33 / 33                    | 11 / 458                         | 3                          | 1 (1 clip)         | 0                         |
| Two classes | 0.50 | 33 / 33                    | 13 / 458                         | 3                          | 0                  | 0                         |
| Two classes | 0.55 | 33 / 33                    | 13 / 458                         | 3                          | 0                  | 0                         |

**The deployment runs the two-class model at `CONF` 0.5**: precision 0.995, recall
0.935, no alert in any of the 130 frames without people, and 33 of 33 clips with people
caught.

**Minimum box size.** `MIN_HEIGHT` drops any person box shorter than a fraction of the
frame height. It is relative to the frame, so it holds at any resolution the camera streams
at. Labelled people in the test frames are at least 7.8% of the frame height, and the
model gives no detection under 7% at `CONF` 0.5, so a limit of **5%** takes nothing
away from the results above and keeps tiny boxes out of the alerts.

`deploy/detect.py` detects both classes, but only people raise an alert: the dog shows
up in the log and in the snapshots of frames with people, and a frame with only the dog
raises nothing. Each box takes its highest-scoring class, as in Ultralytics, so the dog
never comes out as a person just because its person score passes `CONF`. The script also
works with the one-class model, which simply gives no dog boxes.

## License

**Not for commercial use.** The models in this project are trained on CrowdHuman,
which is released for non-commercial research and educational use only. That applies
to the weights and to anything made from them (ONNX and OpenVINO exports included).

- **CrowdHuman**: non-commercial research and education only, under the terms of
  [crowdhuman.org](https://www.crowdhuman.org/).
- **COCO** (dog images of the two-class model): annotations under CC BY 4.0, images
  under the Flickr terms of use.
- **YOLOv8 / Ultralytics**: AGPL-3.0. Commercial use needs an Ultralytics Enterprise
  license ([ultralytics.com/license](https://ultralytics.com/license)).

## Notes

- Tested on Python 3.12.
- The repository keeps two sets of weights: the final model,
  `runs/person/twoclass_stage2_camera/weights/last.pt`, and its INT8 export already in
  `deploy/models/best_openvino_model/`. Everything else under `runs/` is git-ignored and
  stays local.
