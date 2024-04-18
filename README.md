# Learning Motion and Temporal Cues for Unsupervised Video Object Segmentation(Under Review)

> [!NOTE] 
> Thank you for your interest in our work. Currently, our paper is under review, and this repository contains only the test code. We are actively working to prepare the complete codebase, which will include both training and testing phases. We will release the full code soon.

## Demo
<img src="asset/libby.gif" alt="demo1"/> <img src="asset/horsejump-high.gif" alt="demo2"/> <img src="asset/rat.gif" alt="demo2"/>

## Get Started

### Environment

- python == 3.8.15
- torch == 1.10.0
- torchvision == 0.11.0
- cuda == 11.4
- opencv == 4.6.0

### Datasets

Please download the following datasets:

UVOS datasets:

- YouTube-VOS: [YouTube-VOS](https://youtube-vos.org/dataset/)
- DAVIS: [DAVIS](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip)
- YouTube-Objects: [YouTube-Objects](https://data.vision.ee.ethz.ch/cvl/youtube-objects/)
- FBMS: [FBMS](https://lmb.informatik.uni-freiburg.de/resources/datasets/fbms/FBMS_Testset.zip)
- LongVideos: [LongVideos](https://www.kaggle.com/gvclsu/long-videos)

VSOD datasets:

- DAVIS: same as UVOS.
- DAVSOD: [DAVSOD](https://github.com/DengPingFan/DAVSOD)
- SegTrack-V2: [SegTrack-V2](https://github.com/DengPingFan/DAVSOD)
- ViSal: [ViSal](https://github.com/DengPingFan/DAVSOD)

To quickly reproduce our results, we upload the processed data to [Google Drive](https://drive.google.com/drive/folders/1yt4dGuLuhFKpED8TzYr_iWwLrtduMykA?usp=sharing) and [Baidu Disk](https://pan.baidu.com/s/1NkIYp5oJPrPKG8dZLyyBZg)(code: qcbh).

### Models

|    stage    |                          model link                          |
| :---------: | :----------------------------------------------------------: |
|  pre-train  | [Google Drive](https://drive.google.com/drive/folders/1S9St0aRP826Gt9VXPbk9mHGRloNcjpzy?usp=sharing), [Baidu Disk](https://pan.baidu.com/s/1NkIYp5oJPrPKG8dZLyyBZg)(code: qcbh) |
| fine-tuning | [Google Drive](https://drive.google.com/drive/folders/1S9St0aRP826Gt9VXPbk9mHGRloNcjpzy?usp=sharing), [Baidu Disk](https://pan.baidu.com/s/1NkIYp5oJPrPKG8dZLyyBZg)(code: qcbh) |

To reproduct the results we reported in paper, please download the corresponding models and run test script.

### Training

Waiting

### Testing

Download the trained MTNet, and placing it in the `./saves`

```
python test.py [test_model] [task_name] [test_dataset] [output_dir]
```

Testing for UVOS task:

```shell
# Example
# ['DAVIS16', 'YO2SEG', 'FBMS-59', 'LongVideos']
python test.py --test_model ./saves/mtnet.pth --task_name UVOS --test_dataset DAVIS16 --output_dir output
```

Testing for VSOD task:

```shell
# Example
# ['DAVIS16','Easy-35','FBMS-59', 'MCL', 'ViSal', 'SegTrack-V2']
python test.py --test_model ./saves/mtnet.pth --task_name VSOD --test_dataset DAVIS16 --output_dir output
```

### Results

[Precomputed outputs - Google Drive](https://drive.google.com/drive/folders/1N2EInUd4prt87HGme5QoXnz5AdmhQtZH?usp=sharing)

[Precomputed outputs - Baidu Disk](https://pan.baidu.com/s/1NkIYp5oJPrPKG8dZLyyBZg)(code: qcbh)

### Evaluation

Evaluation for UVOS results:

```shell
# Example
python test_scripts/test_for_davis.py --gt_path ../data/DAVIS16/val/mask --result_path output/MTNet/UVOS/DAVIS16/
```

Evaluation for VSOD results:

```python
# Example
python test_scripts/test_vsod/main.py --method MTNet --dataset DAVIS16 --gt_dir test_scripts/test_vsod/gt/ --pred_dir test_scripts/test_vsod/results/
```

### Visualization

Specify the dataset in `visualize.py`, then run:

```python
python visualize.py
```

![](./asset/uvos_vis.jpg)

## References

This repository owes its existence to the exceptional contributions of other projects: 

* STCN: https://github.com/hkchengrex/STCN
* AOT: https://github.com/yoxu515/aot-benchmark
* HFAN: https://github.com/NUST-Machine-Intelligence-Laboratory/HFAN
* FSNet: https://github.com/GewelsJI/FSNet
* AMCNet: https://github.com/isyangshu/AMC-Net
* DAVSOD: https://github.com/DengPingFan/DAVSOD

Many thanks to their invaluable contributions.

