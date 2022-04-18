# PointTrack training/evaluation

PS. Original info was [here](https://github.com/detectRecog/PointTrack).

To train PointTrack model you should have data format like in [KITTI MOTS](http://www.cvlibs.net/datasets/kitti/eval_mots.php). In data folder you can see `kitti_mots.sh` script to automatically download and unpack the data. The structure is following:

```
kitti_mots
│   instances/
│   │    0000/
│   │    0001/
│   │    ...
│   training/
│   │   image_02/
│   │   │    0000/
│   │   │    0001/
│   │   │    ...  
│   testing/
│   │   image_02/
│   │   │    0000/
│   │   │    0001/
│   │   │    ... 
```
### Training

To start training the model you should call:
```bash
# in scripts/tracker/pointtrack/datasets
# at first, process all objects in your dataset
python3 MOTSInstanceMaskPool.py \
  --input_path path_to_the_kitti_mots
  --output_path where_to_save_new_database
  --car to_generate_database_for_cars
  --pedestrian to_generate_database_for_pedestrian

# in scripts/tracker/pointtrack
# at second, start training
python3 train_tracker_with_val.py \
  --config_path path_to_the_config

#see config example at config_mots dir! 
```

### Evaluation

To evaluate, you should download [mots_tools (MOTSA-based metrics)](https://github.com/VisualComputingInstitute/mots_tools) and [TrackEval (HOTA-based metrics)](https://github.com/JonathonLuiten/TrackEval). You should also construct class for matching (see examples in scripts/tracker/pointtrack_class.py). All this evaluators take txt files as an original KITTI MOTS.