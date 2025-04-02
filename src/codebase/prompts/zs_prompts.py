def get_vision_prompts(dataset, prompt_type, arch, fold):
    if dataset.lower() == "waterbirds":
        return get_waterbirds_prompts(prompt_type, arch, fold)
    elif dataset.lower() == "celeba":
        return get_celebA_prompts(prompt_type, arch, fold)
    elif dataset.lower() == "metashift":
        return get_metashift_prompts(prompt_type, arch, fold)


######################################################################
## Change these prompts based on the extracted hypotheses from LLM
######################################################################
def get_waterbirds_prompts(prompt_type, arch, fold):
    if prompt_type.lower() == "baseline" and (arch.lower() == "resnet50" or arch.lower() == "vit"):
        return ["a photo of a landbird"], ["a photo of a waterbird"]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet50" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird on tree_branch",
            "a photo of flying landbird",
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird on top of a tree"
        ], [
            "a photo of waterbird on docks and boats",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying or sitting",
            "a photo of waterbird on oceans and lake",
            "a photo of waterbird on cloudy skies"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_sup_in21k" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird perched on branches",
            "a photo of flying landbird",
            "a photo of landbird inside bamboo forest",
            "a photo of landbird at night"
        ], [
            "a photo of seagul",
            "a photo of waterbird on ocean",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying",
            "a photo of waterbird on beach",
            "a photo of waterbird on boat",
            "a photo of waterbird on rock",
            "a photo of waterbird on docks",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_simclr_in1k" and fold == 0):
        return [
            "a photo of yellow_colored landbird",
            "a photo of landbird on tree_branch",
            "a photo of flying landbird",
            "a photo of landbird inside bamboo forest",
            "a photo of landbird perched on branches",
        ], [
            "a photo of seagul",
            "a photo of duck",
            "a photo of waterbird during sunset",
            "a photo of waterbird on oceans",
            "a photo of waterbird on beach",
            "a photo of waterbird on boat",
            "a photo of waterbird flying or sitting",
            "a photo of waterbird on rock"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_barlow_in1k" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of red landbird ",
            "a photo of landbird on tree_branch",
            "a photo of landbird perched on branches",
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird in the middle of the forest"
        ], [
            "a photo of seagul",
            "a photo of waterbird on oceans",
            "a photo of waterbird during sunset",
            "a photo of waterbird on boats"
            "a photo of waterbird flying",
            "a photo of waterbird on rock"
            "a photo of waterbird on beach",
            "a photo of waterbird on dock"
            "a photo of waterbird on mountains"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_dino_in1k" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird on tree_branch",
            "a photo of flying landbird",
            "a photo of red landbird ",
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird perched on branches",
        ], [
            "a photo of seagul",
            "a photo of waterbird on oceans",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying",
            "a photo of waterbird on rock"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird is on top of a tree",
            "a photo of landbird perched on branches",
            "a photo of landbird in the middle of the forest"
        ], [
            "a photo of waterbird flying or swimming",
            "a photo of waterbird during sunset",
            "a photo of waterbird on ocean or beach"
            "a photo of waterbird on boats or rocks",
            "a photo of waterbird during cloudy or sunny"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_in21k" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird perched on branches",
            "a photo of landbird on wood",
            "a photo of red landbird ",
            "a photo of landbird inside on bamboo forest",
        ], [
            "a photo of waterbird spreading wings",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying",
            "a photo of waterbird on ocean"

        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_oai" and fold == 0):
        return [
            "a photo of landbird in the middle of the forest",
            "a photo of landbird on top of a tree",
            "a photo of flying landbird",
            "a photo of landbird inside on bamboo forest",
            "a photo of yellow_colored landbird ",
            "a photo of red landbird ",
            "a photo of landbird perched on branches"
        ], [
            "a photo of waterbird flying or swimming",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying or sitting",
            "a photo of waterbird on ocean or beach"
            "a photo of waterbird on boats or rocks",
            "a photo of waterbird during cloudy or sunny"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_laion" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird perched on branches",
             "a photo of landbird in the middle of the forest",
            "a photo of landbird perched on trees",
            "a photo of landbird inside on bamboo forest",
        ], [
            "a photo of seagul",
            "a photo of waterbird on oceans",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying over water",
            "a photo of waterbird on rocks"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_swag" and fold == 0):
        return [
            "a photo of yellow_colored landbird ",
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird perched on branches"
             "a photo of landbird in the middle of the forest",
            "a photo of landbird on top of a tree",
            "a photo of red landbird",
        ], [
            "a photo of waterbird catching fish",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying",
            "a photo of waterbird during cloudy weather",
            "a photo of waterbird on oceans",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_dino_in1k" and fold == 0):
        return [
            "a photo of landbird inside on bamboo forest",
            "a photo of landbird inside on tree branches",
            "a photo of landbird perched on branches"
            "a photo of yellow_colored landbird ",
            "a photo of landbird on wood",
            "a photo of flying landbird",
        ], [
            "a photo of seagul",
            "a photo of waterbird during sunset",
            "a photo of waterbird flying",
            "a photo of waterbird on oceans",
            "a photo of waterbird on boats"
        ]

######################################################################
## Change these prompts based on the extracted hypotheses from LLM
######################################################################
def get_celebA_prompts(prompt_type, arch, fold):
    if prompt_type.lower() == "baseline" and (arch.lower() == "resnet50" or arch.lower() == "vit"):
        return ["a photo of a non-blond"], ["a photo of a blond"]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet50" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a red dress",
            "a photo of a blond woman wearing a blue dress",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman with blue eyes"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_sup_in21k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a blue dress",
            "a photo of a blond woman wearing a red dress",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman wearing a black jacket",
            "a photo of a blond woman with long hair",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_simclr_in1k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman with blue eyes",
            "a photo of a blond woman wearing a yellow dress",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a black jacket",
            "a photo of a blond woman wearing a glass",
            "a photo of a blond woman standing in front of a white background",
            "a photo of a blond woman on red carpet",
            "a photo of a blond woman wearing a black dress",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_barlow_in1k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman on red carpet",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman with blue eyes"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_dino_in1k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman on red carpet",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman standing in front of a white background",
            "a photo of a blond woman with blue dress"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_in1k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a black jacket",
            "a photo of a blond woman wearing a red dress",
            "a photo of a blond woman wearing a green dress",
            "a photo of a blond woman wearing a white dress",
            "a photo of a blond woman wearing a white top",
            "a photo of a blond woman wearing a blue dress",
            "a photo of a blond woman wearing a blue top",
            "a photo of a blond woman wearing a red top",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_in21k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman standing on a red carpet",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman with blue eyes",
            "a photo of a blond woman smiling at the camera",
            "a photo of a blond woman standing in front of a white background",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_oai" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman standing on a red carpet",
            "a photo of a blond woman smiling at the camera",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_laion" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman standing on a red carpet",
            "a photo of a blond woman smiling at the camera",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_swag" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a black jacket",
            "a photo of a blond woman standing in front of a white background",
            "a photo of a blond woman wearing a red dress",
            "a photo of a blond woman wearing a red lipstick",
            "a photo of a blond woman standing on a red carpet",
            "a photo of a blond woman with blue eyes",
            "a photo of a blond woman wearing a white dress",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_dino_in1k" and fold == 0):
        return ["a photo of a non-blond"], [
            "a photo of a blond",
            "a photo of a blond woman with long hair",
            "a photo of a blond woman wearing a black dress",
            "a photo of a blond woman standing on a red carpet",
            "a photo of a blond woman wearing a black top",
            "a photo of a blond woman wearing a black jacket",
            "a photo of a blond woman standing in front of a white background",
            "a photo of a blond woman wearing a red dress",
            "a photo of a blond woman wearing a blue dress",
            "a photo of a blond woman wearing a red lipstick",
            "a photo of a blond woman with blue eyes",
            "a photo of a blond woman wearing a white dress",
        ]

######################################################################
## Change these prompts based on the extracted hypotheses from LLM
######################################################################
def get_metashift_prompts(prompt_type, arch, fold):
    if prompt_type.lower() == "baseline" and (arch.lower() == "resnet50" or arch.lower() == "vit"):
        return ["a photo of a dog"], ["a photo of a cat"]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet50" or fold == 0):
        return [
                   "a photo of a dog",
                   "a photo of a dog with an object related to sport",
                   "a photo of a dog on beach",
                   "a photo of a dog in motion",
                   "a photo of a dog with objects in their mouths",
                   "a photo of a dog on leashes",
               ], [
                   "a photo of a cat",
                   "a photo of a cat on laptops"
                   "a photo of a cat on bathroom",
                   "a photo of a cat on beds",
                   "a photo of a cat on desks",
                   "a photo of a cat on sinks",
                   "a photo of a sitting cat"
               ]
    
######################################################################
## Change these prompts based on the extracted hypotheses from LLM
######################################################################
def get_nih_prompts(prompt_type, arch, fold):
    if prompt_type.lower() == "baseline" and (arch.lower() == "resnet50" or arch.lower() == "vit"):
        return ["no pneumothorax"], ["pneumothorax"]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet50" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with chest tubes",
            "pneumothorax with fluid levels", "pneumothorax with varying size and description",
            "pneumothorax affecting different sides of the body (right/left)"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_sup_in21k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with postoperative changes",
            "pneumothorax with fluid levels",
            "pneumothorax with varying air-fluid levels",
            "apical pneumothorax"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_simclr_in1k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with pleural effusion",
            "pneumothorax with fluid levels",
            "pneumothorax with the presence of pleural thickening",
            "pneumothorax with postoperative changes"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_barlow_in1k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with chest tubes",
            "pneumothorax with fluid levels",
            "pneumothorax with varying size and distribution",
            "pneumothorax affecting different sides of the body (right/left)"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "resnet_dino_in1k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with postoperative changes"
            "pneumothorax with fluid levels",
            "pneumothorax with pleural effusion",
            "pneumothorax with lateral view mentions"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_in1k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with chest tubes",
            "pneumothorax with fluid levels"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_in21k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with varying size and distribution",
            "pneumothorax with chest tubes and drains",
            "pneumothorax with fluid levels",
            "pneumothorax with the specific anatomical location descriptions (e.g., apical, basal, lateral)"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_oai" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with pleural effusion",
            "pneumothorax with chest tubes",
            "pneumothorax with air fluid levels",
            "pneumothorax with the apical or basal location"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_clip_laion" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with postoperative changes"
            "pneumothorax with chest tubes",
            "pneumothorax with fluid levels",
            "pneumothorax with pleural effusion",
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_sup_swag" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with varying size and description",
            "pneumothorax with chest tubes",
            "pneumothorax with fluid and air levels",
            "pneumothorax affecting the left side"
        ]
    elif prompt_type.lower() == "slice" and (arch.lower() == "vit_dino_in1k" and fold == 0):
        return ["no pneumothorax"], [
            "pneumothorax",
            "loculated characteristics of pneumothorax",
            "pneumothorax with chest tubes",
            "pneumothorax with fluid levels", "pneumothorax with varying size and volume",
            "pneumothorax affecting the location of the body (right/left)"
        ]

######################################################################
## Change these prompts based on the extracted hypotheses from LLM
######################################################################
def get_rsna_prompts(prompt_type, arch, fold, dataset):
    if dataset.lower() == "rsna":
        if prompt_type.lower() == "baseline":
            return ["no cancer", "no malignancy"], ["cancer", "malignancy"]
        elif prompt_type.lower() == "slice" and fold == 0:
            return ["no cancer"], [
                "cancer",
                "cancer with scattered calcifications",
                "cancer with bilateral occurrences",
                "cancer with multiple densities",
                "cancer with vascular calcifications",

                "malignancy",
                "malignancy with scattered calcifications",
                "malignancy with bilateral occurrences",
                "malignancy with multiple densities",
                "malignancy with vascular calcifications",
            ]
    elif dataset.lower() == "vindr":
        if prompt_type.lower() == "baseline":
            return ["no cancer", "no malignancy"], ["cancer", "malignancy"]
        elif prompt_type.lower() == "slice" and fold == 0:
            return ["no cancer"], [
                "cancer",
                "cancer with calcifications",
                "cancer with postsurgical changes",
                "cancer with benign appearing nodules",
                "cancer with implants",

                "malignancy",
                "malignancy with calcifications",
                "malignancy with postsurgical changes",
                "malignancy with benign appearing nodules",
                "malignancy with implants",
            ]