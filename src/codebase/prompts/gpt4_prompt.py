def create_NIH_prompts(content):
    prompt_dict = """
    prompt_dict = {
        "H1_<attribute>": [List of prompts],
        "H2_<attribute>": [List of prompts]
        ...
    }

    """

    hyp_dict = """
        hypothesis_dict = {
        "H1": "The classifier is making mistake  as it is biased toward <attribute>",
        "H2": "The classifier is making mistake as it is biased toward <attribute>",
        "H3": "The classifier is making mistake as it is biased toward <attribute>",
        ...
        }
    """

    prompt = f"""
    Context: Pneumothorax classification from chest-x-rays using a deep neural network
    Analysis post training: On a validation set, 
    a. Get the difference between the image embeddings of correct and incorrectly classified samples to estimate the features present in the correctly classified samples but missing in the misclassified samples.
    b. Retrive the top 50 sentences from radiology report that matches closely to the embedding difference in step a.
    c. The sentence list is given below:
    {content}
    
    Task: 
    Ignore '___' as they are due to anonymization.
    Consider the consistent attributes present in the descriptions of correctly classified and misclassified samples regarding the positive Pneumothorax patients. Formulate hypotheses based on these attributes. Attributes are all the concepts (e.g, explicit or implicit anatomies, observations, or any concept leading to potential bias) in the sentences other than the class label (Pneumothorax in this case). Assess how these characteristics might be influencing the classifier's performance. Your response should contain only the list of top hypothesis, nothing else. For the response, you should be the following python dictionary template, no extra sentence:
    {hyp_dict}
    
    To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images where the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the class label "Pneumothorax" (i.e, positive Pneumothorax cases)
    
    Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:
    
    {prompt_dict}
    
    Each attribute hypothesis should contain 5 prompts.
    
    So final response should follow the below format strictly (nothing else, no extra sentence): 
    ```python
        hypothesis_dict 
        prompt_dict 
    ```
    """

    return prompt


def create_RSNA_prompts(content):
    prompt_dict = """
    prompt_dict = {
        "H1_<attribute>": [List of prompts],
        "H2_<attribute>": [List of prompts]
        ...
    }

    """

    hyp_dict = """
        hypothesis_dict = {
        "H1": "The classifier is making mistake  as it is biased toward <attribute>",
        "H2": "The classifier is making mistake as it is biased toward <attribute>",
        "H3": "The classifier is making mistake as it is biased toward <attribute>",
        ...
        }
    """

    prompt = f"""
    Context: Breast cancer classification from mammograms using a deep neural network
    Analysis post training: On a validation set, 
    a. Get the difference between the image embeddings of correct and incorrectly classified samples to estimate the features present in the correctly classified samples but missing in the misclassified samples.
    b. Retrive the top 50 sentences from radiology report that matches closely to the embedding difference in step a.
    c. The sentence list is given below:
    {content}
    
    Task: 
    Ignore '___' as they are due to anonymization.
    Consider the consistent attributes present in the descriptions of correctly classified and misclassified samples regarding the positive cancer patients. Formulate hypotheses based on these attributes. Attributes are all the concepts (e.g, explicit or implicit anatomies, observations, any symptom of change related to the disease, demography related information or any concept leading to potential bias) in the sentences other than the class label (Cancer in this case). Assess how these characteristics might be influencing the classifier's performance. Your response should contain only the list of top hypothesis, nothing else. For the response, you should be the following python dictionary template, no extra sentence:
    {hyp_dict}
    
    To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images where the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the class label "Cancer" (i.e, positive cancer cases)
    
    Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:
    
    {prompt_dict}
    
    Each attribute hypothesis should contain 5 prompts.
    
    So final response should follow the below format strictly (nothing else, no extra sentence): 
    ```python
        hypothesis_dict 
        prompt_dict 
    ```
    """

    return prompt


def create_CELEBA_prompts(content):
    prompt_dict = """
    prompt_dict = {
        "H1_<attribute>": [List of prompts],
        "H2_<attribute>": [List of prompts]
        ...
    }

    """

    hyp_dict = """
        hypothesis_dict = {
        "H1": "The classifier is making mistake  as it is biased toward <attribute>",
        "H2": "The classifier is making mistake as it is biased toward <attribute>",
        "H3": "The classifier is making mistake as it is biased toward <attribute>",
        ...
        }
    """

    prompt = f"""
    Context: Hair color (Blond/Non blond) classification from images of person using a deep neural network
    Analysis post training: On a validation set for the class label "Blond", 
    a. Get the difference between the image embeddings of correct and incorrectly classified samples.
    b. Retrieve the top K sentences from the captions of the images that matches closely to the embedding difference in step a.
    c. The sentence list is given below in the descending order of similarity with the embedding difference:
    {content}
    
    These sentences represent the features present in the correctly classified samples but missing in the misclassified samples.
    Task: 
    The task is to reason why the model is making mistakes on the misclassified samples based on the sentences. 
    To do so, consider the attributes present in the above captions regarding the samples with Blond hair (i.e, the class label "Blond"). So come up with the list of hypotheses based based on all these attributes to reason why a model makes systematic mistakes. For the hypotheses, you should be the following python dictionary template, no extra sentence:
    {hyp_dict}
    You must follow the following rules to construct the hypotheses:
    1. You must pick specific attributes, e.g, white dress, not generic attributes like dress color.
    2. Your hypotheses must be based on the attributes present in the captions, nothing else.
    3. You must pay close attention to the attributes that are consistently present in the sentences. These attributes are likely to be the cause of the systematic mistakes on the misclassified samples.
    4. You must construct as many hypotheses possible.
    
    Next you have to test the hypothesis. To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images for which the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the class label "Blond" (i.e, men/women with Blond hair)
    
    Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:
    
    {prompt_dict}
    
    Each attribute hypothesis should contain 5 prompts.
    
    So final response should follow the below format strictly (nothing else, no extra sentence): 
    ```python
        hypothesis_dict 
        prompt_dict 
    ```
    """

    return prompt


def create_Waterbirds_prompts(content):
    prompt_dict = """
    prompt_dict = {
        "H1_<attribute>": [List of prompts],
        "H2_<attribute>": [List of prompts]
        ...
    }

    """

    hyp_dict = """
        hypothesis_dict = {
        "H1": "The classifier is making mistake  as it is biased toward <attribute>",
        "H2": "The classifier is making mistake as it is biased toward <attribute>",
        "H3": "The classifier is making mistake as it is biased toward <attribute>",
        ...
        }
    """

    prompt = f"""
    Context: Bird classification from images using a deep neural network
    Analysis post training: On a validation set, 
    a. Get the difference between the image embeddings of correct and incorrectly classified samples.
    b. Retrieve the top K sentences from the captions of the images that matches closely to the embedding difference in step a.
    c. The sentence list is given below in the descending order of similarity with the embedding difference:
    {content}
    
    These sentences represent the features present in the correctly classified samples but missing in the misclassified samples.
    Task: 
    The task is to reason why the model is making mistakes on the misclassified samples based on the sentences. To do so, consider the attributes present in the above captions regarding to the specific bird species. Attributes are all the concepts other than the class label. So come up with the list of hypotheses based based on these attributes to reason why a model makes systematic mistakes. For the hypotheses, you should be the following python dictionary template, no extra sentence:
    {hyp_dict}
    You must follow the following rules to construct the hypotheses:
    1. You must pick specific attributes, e.g, blue, not generic attributes like color.
    2. Your hypotheses must be based on the attributes present in the captions, nothing else.
    3. You must pay close attention to the attributes that are consistently present in the sentences. These attributes are likely to be the cause of the systematic mistakes on the misclassified samples.
    4. You must construct as many hypotheses possible.
    
    Next you have to test the hypothesis. To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images for which the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the specific bird.
    
    Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:
    
    {prompt_dict}
    
    Each attribute hypothesis should contain 5 prompts.
    
    So final response should follow the below format strictly (nothing else, no extra sentence): 
    ```python
        hypothesis_dict 
        prompt_dict 
    ```
    """

    return prompt


def create_Metashift_prompts(content):
    prompt_dict = """
    prompt_dict = {
        "H1_<attribute>": [List of prompts],
        "H2_<attribute>": [List of prompts]
        ...
    }

    """

    hyp_dict = """
        hypothesis_dict = {
        "H1": "The classifier is making mistake  as it is biased toward <attribute>",
        "H2": "The classifier is making mistake as it is biased toward <attribute>",
        "H3": "The classifier is making mistake as it is biased toward <attribute>",
        ...
        }
    """

    cat_prompt = f"""
    Context: Cat vs Dog classification from images using a deep neural network
    Analysis post training: On a validation set, 
    a. Get the difference between the image embeddings of correct and incorrectly classified samples.
    b. Retrieve the top K sentences from the captions of the images that matches closely to the embedding difference in step a.
    c. The sentence list is given below in the descending order of similarity with the embedding difference:
    {content}

    These sentences represent the features present in the correctly classified samples but missing in the misclassified samples.
    Task: 
    The task is to reason why the model is making mistakes on the misclassified samples based on the sentences for the class label 'cat'. To do so, consider the attributes present in the above captions regarding to the specific bird species. Attributes are all the concepts other than the class label (i.e, cat). So come up with the list of hypotheses based based on these attributes to reason why a model makes systematic mistakes. For the hypotheses, you should be the following python dictionary template, no extra sentence:
    {hyp_dict}
    You must follow the following rules to construct the hypotheses:
    1. You must pick specific attributes, e.g, blue, not generic attributes like color.
    2. Your hypotheses must be based on the attributes present in the captions, nothing else.
    3. You must pay close attention to the attributes that are consistently present in the sentences. These attributes are likely to be the cause of the systematic mistakes on the misclassified samples.
    4. You must construct as many hypotheses possible.

    Next you have to test the hypothesis. To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images for which the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the specific bird.

    Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:

    {prompt_dict}

    Each attribute hypothesis should contain 5 prompts.

    So final response should follow the below format strictly (nothing else, no extra sentence): 
    ```python
        hypothesis_dict 
        prompt_dict 
    ```
    """

    dog_prompt = f"""
        Context: Cat vs Dog classification from images using a deep neural network
        Analysis post training: On a validation set, 
        a. Get the difference between the image embeddings of correct and incorrectly classified samples.
        b. Retrieve the top K sentences from the captions of the images that matches closely to the embedding difference in step a.
        c. The sentence list is given below in the descending order of similarity with the embedding difference:
        {content}

        These sentences represent the features present in the correctly classified samples but missing in the misclassified samples.
        Task: 
        The task is to reason why the model is making mistakes on the misclassified samples based on the sentences for the class label 'dog'. To do so, consider the attributes present in the above captions regarding to the specific bird species. Attributes are all the concepts other than the class label (i.e, dog). So come up with the list of hypotheses based based on these attributes to reason why a model makes systematic mistakes. For the hypotheses, you should be the following python dictionary template, no extra sentence:
        {hyp_dict}
        You must follow the following rules to construct the hypotheses:
        1. You must pick specific attributes, e.g, blue, not generic attributes like color.
        2. Your hypotheses must be based on the attributes present in the captions, nothing else.
        3. You must pay close attention to the attributes that are consistently present in the sentences. These attributes are likely to be the cause of the systematic mistakes on the misclassified samples.
        4. You must construct as many hypotheses possible.

        Next you have to test the hypothesis. To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images for which the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the specific bird.

        Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:

        {prompt_dict}

        Each attribute hypothesis should contain 5 prompts.

        So final response should follow the below format strictly (nothing else, no extra sentence): 
        ```python
            hypothesis_dict 
            prompt_dict 
        ```
        """

    return cat_prompt, dog_prompt