# paddleOCR_data_preparation
Full process for data creation and checking:

#### text detection data creation process:

1. to create combined data from multiple folders: change necessary dirs <br>
python perfolder_anno_to_combined.py

2. split the combined data into train-test: change necessary dirs <br>
python train_test_split.py

3. to check annotaions after spliting: change necessary dirs <br>
python text_annotation_plotting.py

3. a: if you want to create annotations for yolo and fasterrcnn<br>
python paddle2yolofaster.py

#### text recognition data creation process
4. to create text recognition data: change necessary dirs <br>
python recog_data_creation.py

5. to check text annotation: change necessary dirs <br>
python recog_data_check.py
