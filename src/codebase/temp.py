import pickle

_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/RSNA/fold0/aucroc0.89/clip_img_encoder_tf_efficientnet_b5_ns-detect/cancer_prompt_dict.pkl"
with open(_path, 'rb') as file:
    my_object = pickle.load(file)
print(my_object)

# _dict = {'H1_vascular calcifications': ['The image shows vascular calcifications.',
#                                         'Vascular calcifications are present in the mammogram.',
#                                         'This mammogram contains vascular calcifications.',
#                                         'Identify the presence of vascular calcifications in the image.',
#                                         'Vascular calcifications are evident in this image.'],
#          'H2_scattered calcifications': ['The image shows scattered calcifications.',
#                                          'Scattered calcifications are present in the mammogram.',
#                                          'This mammogram contains scattered calcifications.',
#                                          'Identify the presence of scattered calcifications in the image.',
#                                          'Scattered calcifications are evident in this image.'],
#          'H3_benign appearing calcifications': ['The image shows benign appearing calcifications.',
#                                                 'Benign appearing calcifications are present in the mammogram.',
#                                                 'This mammogram contains benign appearing calcifications.',
#                                                 'Identify the presence of benign appearing calcifications in the image.',
#                                                 'Benign appearing calcifications are evident in this image.']}
#
# with open(_path, 'wb') as file:
#     pickle.dump(_dict, file)
