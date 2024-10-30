import pickle

_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/blonde_prompt_dict.pkl"
with open(_path, 'rb') as file:
    my_object = pickle.load(file)
print(my_object)

_dict = {
    'H1_smiling expressions': ['A person with blonde hair not smiling',
                               'A person with blonde hair with a neutral expression',
                               'A person with blonde hair frowning', 'A person with blonde hair looking serious',
                               'A person with blonde hair with a sad expression'],
    'H2_light-colored backgrounds': ['A person with blonde hair against a dark background',
                                     'A person with blonde hair against a colorful background',
                                     'A person with blonde hair against a textured wall',
                                     'A person with blonde hair against a nature background',
                                     'A person with blonde hair against a patterned backdrop'],
    'H3_wearing earrings': ['A person with blonde hair not wearing earrings',
                            'A person with blonde hair with no jewelry', 'A person with blonde hair with bare ears',
                            'A person with blonde hair without accessories',
                            'A person with blonde hair with no visible earrings'],
    'H4_wearing makeup': ['A person with blonde hair without makeup', 'A person with blonde hair with a natural face',
                          'A person with blonde hair with no visible makeup',
                          'A person with blonde hair with a bare face',
                          'A person with blonde hair with minimal makeup'],
    'H5_posing against event backdrops': ['A person with blonde hair in a casual setting',
                                          'A person with blonde hair in a home environment',
                                          'A person with blonde hair in a natural outdoor setting',
                                          'A person with blonde hair in a non-event location',
                                          'A person with blonde hair in a personal space'],
    'H6_wavy hair': ['A person with blonde hair that is straight', 'A person with blonde hair that is curly',
                     'A person with blonde hair in a ponytail', 'A person with blonde hair in a bun',
                     'A person with blonde hair that is braided'],
    'H7_wearing light-colored tops': ['A person with blonde hair wearing a dark top',
                                      'A person with blonde hair wearing a patterned top',
                                      'A person with blonde hair wearing a colorful top',
                                      'A person with blonde hair wearing a black top',
                                      'A person with blonde hair wearing a red top'],
    'H8_having blue eyes': ['A person with blonde hair with brown eyes', 'A person with blonde hair with green eyes',
                            'A person with blonde hair with hazel eyes', 'A person with blonde hair with gray eyes',
                            'A person with blonde hair with dark eyes'],
    'H9_voluminous hair': ['A person with blonde hair that is flat', 'A person with blonde hair that is thin',
                           'A person with blonde hair that is sleek', 'A person with blonde hair that is tied back',
                           'A person with blonde hair that is short'],
    'H10_posing indoors': ['A person with blonde hair outdoors', 'A person with blonde hair in a park',
                           'A person with blonde hair on a beach', 'A person with blonde hair in a garden',
                           'A person with blonde hair in a forest'],
    "H11_women": [
        "A woman with blonde hair",
        "A female with blonde hair",
        "A girl with blonde hair",
        "A blonde-haired woman smiling",
        "A woman with long blonde hair"
    ]
}

with open(_path, 'wb') as file:
    pickle.dump(_dict, file)

_path = "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/CelebA/resnet_sup_in1k_attrNo/CelebA_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/blonde_hypothesis_dict.pkl"
with open(_path, 'rb') as file:
    my_object = pickle.load(file)
print(my_object)

_dict = {
    'H1': 'The classifier is making mistake as it is biased toward smiling expressions',
    'H2': 'The classifier is making mistake as it is biased toward light-colored backgrounds',
    'H3': 'The classifier is making mistake as it is biased toward wearing earrings',
    'H4': 'The classifier is making mistake as it is biased toward wearing makeup',
    'H5': 'The classifier is making mistake as it is biased toward posing against event backdrops',
    'H6': 'The classifier is making mistake as it is biased toward wavy hair',
    'H7': 'The classifier is making mistake as it is biased toward wearing light-colored tops',
    'H8': 'The classifier is making mistake as it is biased toward having blue eyes',
    'H9': 'The classifier is making mistake as it is biased toward voluminous hair',
    'H10': 'The classifier is making mistake as it is biased toward posing indoors',
    "H11": "The classifier is making mistake as it is biased toward images of women",
}

with open(_path, 'wb') as file:
    pickle.dump(_dict, file)
