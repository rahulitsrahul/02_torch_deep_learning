import json
import pandas as pd

class parse_annotation_file(object):
    def __init__(self, annotation_file_path, im_ext):
        self.annotation_file_path = annotation_file_path
        file = open(annotation_file_path, encoding="utf-8")
        json_data = json.load(file)
        file.close()
        # data = json_data['_via_img_metadata']
        self.im_ext = im_ext
        self.data = json_data
        self.df = pd.DataFrame(columns=["image_id", "x1", "y1", "x2", "y2", "bbox_id"])
        self.update_df()
        
    def update_df(self):
        index=0
        elements = list(self.data.keys())
        classes = ['tube', 'cap'] # Which classes need to be added to the dataFrame
        for el in elements:
            cur_el = self.data[el]
            file_name= cur_el['filename']
            regions = cur_el['regions']
            
            for region in regions:
                label = region['region_attributes']['labels']
                if label in classes:
                    x = region['shape_attributes']['x']
                    y = region['shape_attributes']['y']
                    width = region['shape_attributes']['width']
                    height = region['shape_attributes']['height']
              
                    if(label == 'tube'):
                        label_value = 64
                        id_val = 1
                    elif(label =='cap'):
                        label_value = 128
                        id_val = 2
                    # elif(label == 'umpire'):
                    #     label_value = 255
                    #     id_val = 3
                    
                    image_id = file_name.split(self.im_ext)[0]
                    x1 = x
                    y1 = y
                    x2 = x1 + width
                    y2 = y1 + height
                    
                    self.df.loc[index] = [image_id, x1, y1, x2, y2, id_val]
                    index += 1
        