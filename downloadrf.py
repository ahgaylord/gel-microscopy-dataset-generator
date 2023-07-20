from roboflow import Roboflow
import os
import shutil
from model.JSON2YOLO import general_json2yolo as j2y

my_path = os.path.abspath(os.path.dirname(__file__))
target = my_path + "\\model\\datasets\\smith_vid_COCO"
my_key_path = my_path + "\\SECRET\\myroboflowkey.txt"

key_file = open(my_key_path)
my_key = key_file.read()

rf = Roboflow(api_key=my_key)
project = rf.workspace("gel-machine-learning").project("gel-detection")
dataset = project.version(3).download("coco", location=target, overwrite=True)

coco_training = target + "\\train"
coco_validate = target + "\\valid"
coco_test = target + "\\test"

j2ydir = my_path + "\\new_dir"
j2ydir_lbls = my_path + "\\new_dir\\labels"
j2ydir_imgs = my_path + "\\new_dir\\images"

training = my_path + "\\model\\datasets\\smith_vid\\training"
validate = my_path + "\\model\\datasets\\smith_vid\\valid"
test = my_path + "\\model\\datasets\\smith_vid\\test"

j2y.convert_coco_json(json_dir=coco_training, use_segments=True)

shutil.move(j2ydir_lbls, training)
shutil.move(j2ydir_imgs, training)

# shutil.rmtree(j2ydir_lbls)
# shutil.rmtree(j2ydir_imgs)
shutil.rmtree(j2ydir)

j2y.convert_coco_json(json_dir=coco_validate, use_segments=True)

shutil.move(j2ydir_lbls, validate)
shutil.move(j2ydir_imgs, validate)

# shutil.rmtree(j2ydir_lbls)
# shutil.rmtree(j2ydir_imgs)
shutil.rmtree(j2ydir)

j2y.convert_coco_json(json_dir=coco_test, use_segments=True)

shutil.move(j2ydir_lbls, test)
shutil.move(j2ydir_imgs, test)

# shutil.rmtree(j2ydir_lbls)
# shutil.rmtree(j2ydir_imgs)
shutil.rmtree(j2ydir)


