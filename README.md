# Select, Label, and Mix (SLM)

The repository contains the codes for the paper "Select, Label, and Mix: Learning Discriminative Invariant Feature Representations for Partial Domain Adaptation" part of Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2023.

[Aadarsh Sahoo<sup>1</sup>](https://aadsah.github.io/), [Rameswar Panda<sup>1</sup>](https://rpand002.github.io/), [Rogerio Feris<sup>1</sup>](https://www.rogerioferis.org/), [Kate Saenko<sup>1,2</sup>](http://ai.bu.edu/ksaenko.html), [Abir Das<sup>3</sup>](https://cse.iitkgp.ac.in/~adas/)

<sup>1</sup> MIT-IBM Watson AI Lab, <sup>2</sup> Boston University, <sup>3</sup> IIT Kharagpur

[[Paper]](https://openaccess.thecvf.com/content/WACV2023/papers/Sahoo_Select_Label_and_Mix_Learning_Discriminative_Invariant_Feature_Representations_for_WACV_2023_paper.pdf) [[Project Page]](https://cvir.github.io/projects/slm)


### Preparing the Environment

#### Conda 
Please use the `slm_environment.yml` file to create the conda environment `SLM` as:

```
conda env create -f slm_environment.yml
```

#### Pip
Please use the `requirements.txt` file to install all the required dependencies as:

```
pip install -r requirements.txt
```

### Data Directory Structure
All the datasets should be stored in the folder `./data` following the convention `./data/<dataset_name>/<domain_names>`. E.g. for `Office31` the structure would be as follows:

```
    .
    ├── ...
    ├── data
    │   ├── Office31
    │   │    ├── amazon
    │   │    ├── webcam
    │   │    ├── dslr
    │   └── ...
    └── ...
```

For using datasets stored in some other directories, please update the path to the data accordingly in the txt files inside the folder `./data_labels`.

The official download links for the datasets used for this paper are:

**Office31**: https://people.eecs.berkeley.edu/~jhoffman/domainadapt/#datasets_code

**OfficeHome**: http://hemanthdv.org/OfficeHome-Dataset/

**ImageNet-Caltech**: http://www.image-net.org/, http://www.vision.caltech.edu/Image_Datasets/Caltech256/

**VisDA-2017**: http://ai.bu.edu/visda-2017/#download

### Training SLM
Here is a sample and recomended command to train SLM for the transfer task of `Amazon -> Webcam` from `Office31` dataset:

```
CUDA_VISIBLE_DEVICES=0 python main.py --manual_seed 1 --dataset_name Office31 --src_dataset amazon --tgt_dataset webcam  --batch_size 64 --model_root ./checkpoints_a31_w10 --save_in_steps 500 --log_in_steps 10 --eval_in_steps 10 --model_name resnet50 --classifier_name resnet50 --source_images_path ./data_labels/Office31/amazon_31_list.txt --target_images_path ./data_labels/Office31/webcam_10_list.txt --pseudo_threshold 0.3 --warmstart_models True --num_iter_adapt 10000 --num_iter_warmstart 5000 --learning_rate 0.0005 --learning_rate_ws 0.001
```

For detailed description regarding the arguments, use:

```
python main.py --help
```

### Citing SLM

If you use codes in this repository, consider citing SLM. Thanks!

```
@inproceedings{sahoo2023select,
  title={Select, label, and mix: Learning discriminative invariant feature representations for partial domain adaptation},
  author={Sahoo, Aadarsh and Panda, Rameswar and Feris, Rogerio and Saenko, Kate and Das, Abir},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4210--4219},
  year={2023}
}
```