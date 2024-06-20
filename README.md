# paddleOCR_data_preparation
Full process for data creation and checking:

#### text detection data creation process:

1. to create combined data from multiple folders: change necessary dirs
python perfolder_anno_to_combined.py

2. split the combined data into train-test: change necessary dirs
python train_test_split.py

3. to check annotaions after spliting: change necessary dirs
python text_annotation_plotting.py

#### text recognition data creation process
4. to create text recognition data: change necessary dirs
python recog_data_creation.py

5. to check text annotation: change necessary dirs
python recog_data_check.py
