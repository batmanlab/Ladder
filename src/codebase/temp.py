# import pickle
#
# _path = "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip_mc/pneumothorax_prompt_dict.pkl"
# with open(_path, 'rb') as file:
#     my_object = pickle.load(file)
# print(my_object)
#
# _dict = {
#     "H1_chest tubes": [
#         "Chest x-ray showing a pneumothorax with multiple chest tubes in place.",
#         "Image of a pneumothorax with a recently inserted chest tube.",
#         "Pneumothorax present, chest tube placement for drainage.",
#         "Radiograph highlighting the presence of multiple chest tubes and a pneumothorax.",
#         "A pneumothorax case with chest tube placement for therapeutic intervention."
#     ],
#     "H2_size and location": [
#         "Large right-sided pneumothorax visible in the chest x-ray.",
#         "Chest x-ray showcasing a small apical pneumothorax.",
#         "Image depicting a loculated pneumothorax in the right lung base.",
#         "Basilar pneumothorax evident in the chest radiograph.",
#         "Pneumothorax located in the left upper lobe, small in size."
#     ],
#     "H3_pleural effusion": [
#         "Chest x-ray exhibiting a pneumothorax with a significant pleural effusion.",
#         "Image showing a pneumothorax and associated hydropneumothorax.",
#         "Pneumothorax accompanied by a moderate pleural effusion.",
#         "Radiograph highlighting the presence of both pneumothorax and pleural fluid.",
#         "A case of pneumothorax with a substantial amount of pleural effusion present."
#     ],
#     "H4_interval changes": [
#         "Chest x-ray showing interval decrease in pneumothorax size.",
#         "Image depicting a significant increase in pneumothorax size since the last examination.",
#         "Pneumothorax with little to no change observed over time.",
#         "Radiograph highlighting an interval increase in pleural fluid, decreasing pneumothorax.",
#         "A case of pneumothorax showing improvement with decreased air over time."
#     ],
#     "H5_multi-loculated": [
#         "Complex, multi-loculated pneumothorax observed in the chest x-ray.",
#         "Image of a pneumothorax with multiple fluid-filled pockets.",
#         "Pneumothorax featuring several loculated air collections.",
#         "Radiograph demonstrating a challenging case of multi-loculated pneumothorax.",
#         "A pneumothorax with a complex pattern of loculations and fluid levels."
#     ]
# }
#
# with open(_path, 'wb') as file:
#     pickle.dump(_dict, file)
#
# _path = "/restricted/projectnb/batmanlab/shawn24/PhD/Ladder/out/NIH/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip_mc/pneumothorax_hypothesis_dict.pkl"
# with open(_path, 'rb') as file:
#     my_object = pickle.load(file)
# print(my_object)
#
# _dict = {
#     "H1": "The classifier is making mistakes as it is biased toward the presence of chest tubes.",
#     "H2": "The classifier is making mistakes as it is biased toward the size and location of the pneumothorax (e.g., basilar, apical, loculated).",
#     "H3": "The classifier is making mistakes as it is biased toward the presence of associated pleural effusion or hydropneumothorax.",
#     "H4": "The classifier is making mistakes as it is biased toward the change in pneumothorax size over time (interval changes).",
#     "H5": "The classifier is making mistakes due to the presence of complex, multi-loculated pneumothoraces."
# }
#
# with open(_path, 'wb') as file:
#     pickle.dump(_dict, file)


import base64
import vertexai
from vertexai.generative_models import GenerativeModel, SafetySetting, Part


def multiturn_generate_content():
    vertexai.init(project="gen-lang-client-0586677956", location="us-central1")
    model = GenerativeModel(
        "gemini-1.5-flash-002",
    )
    chat = model.start_chat()
    response = chat.send_message(
        [text1_1],
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(response)
    clean_python_code = response.text.strip('`').split('python\n')[1]
    clean_python_code = clean_python_code.split('```')[0]  # Remove the ending ```

    local_vars = {}
    exec(clean_python_code, {}, local_vars)
    hypothesis_dict = local_vars.get('hypothesis_dict')
    prompt_dict = local_vars.get('prompt_dict')
    # Now the dictionaries hypothesis_dict and prompt_dict are available
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)


text1_1 = """Context: Pneumothorax classification from chest-x-rays using a deep neural network
Analysis post training: On a validation set,
  a. Get the difference between the image embeddings of correct and incorrectly classified samples to estimate the features present in the correctly classified samples but missing in the misclassified samples.
  b. Retrive the top 50 sentences from radiology report that matches closely to the embedding difference in step a.
  c. The sentence list is given below:
  1. perhaps mild increase in hydropneumothorax but with chest tube remaining in place and no striking change
2. in comparison with the study of ___ , there is little change in the 3 left chest tubes with area of hydro pneumothorax persisting in the lateral aspect of the upper left chest as well as probably the left lung base
3. a moderate sized loculated hydropneumothorax demonstrates decrease in fluid component and increasing gas component , particularly in the right base
4. small right pleural effusion has replaced the previous basal pneumothorax that developed with previous drainage of pleural effusion and placement of 2 thoracostomy tubes
5. 2 right indwelling pleural drains are unchanged in their respective positions , and there has probably been some decrease in the volume of the right posterior air and pleural collection in the rib right lower hemi thorax
6. interval placement of right apical and right base pleural drains with slight decrease in right hydropneumothorax
7. other less likely possibility include expansion of known loculated hydropneumothorax ( chest tube does not appear to be draining this region )
8. increasing fluid within the multiple pockets of the pneumothorax on the right
9. decreased fluid and increased air in the right basilar hydropneumothorax , where the pleural catheter resides
10. moderate right pleural effusion with loculated hydro pneumothorax components is again demonstrated , with apparent slight increase in extent of right basilar hydro pneumothorax
11. three right - sided chest tubes remain in place with decrease in the loculated basilar hydropneumothoraces and some interval improvement in aeration at the right base
12. loculated right hydro pneumothorax , with right basilar pneumothorax component slightly increased
13. the only change is increasing fluid within a loculated hydropneumothorax , with corresponding decrease in air , located along the left lateral chest wall , of uncertain significance in the short - term postoperative course
14. multiple small loculated hydropneumothoraces are again demonstrated , with interval worsening of loculated hydropneumothoraces at the left base
15. fluid has now replaced air in the lateral component of the multi loculated left hydro pneumothorax , despite the insertion in that location of a new small drainage catheter
16. loculated he mall / hydro pneumothorax in the upper portion of the chest as well as in the left lung base are unchanged
17. small - to - moderate right hydropneumothorax , increase in the lateral costal fluid collection , and stable small apical air component
18. successful placement of chest tube and pleurx tube , small right basal loculated pneumothorax replaces area of successful pleural drainage
19. interval insertion of a right - sided chest tube in good position , right - sided hydro pneumothorax has changed with increased pleural air are and relative decrease of the pleural fluid
20. in comparison with the study of ___ , there is little overall change in the loculated hydro pneumothorax at the right base with chest tube in place
21. one of two right - sided pleural tubes has been removed in the interval , with no substantial change in appearance of loculated hydropneumothoraces at the right lung apex
22. there is no appreciable interval change in the loculated right hydropneumothorax other than a slight increase in the fluid component
23. new small right basal pneumothorax absent a fluid level , suggests that the fluid and air are loculated separately
24. right - sided loculated hydropneumothorax has increased
25. interval increase in right loculated hydropneumothorax
26. right hydropneumothorax persists with now more fluid in the lateral basilar portion with air collecting at the apex
27. persistent hydropneumothorax , however , no larger than prior , status post chest tube clamping
28. since recent radiograph from earlier the same date , 2 chest tubes have been removed from the right hemi thorax with development of a right basilar hydropneumothorax as well as a persistent loculated fluid collection at the right apex
29. stable , loculated , moderate right hydro pneumothorax and moderate dependent pleural effusion
30. interval increase in the loculated left basal pneumothorax
31. slight interval increase in known right basilar hydropneumothorax following chest tube clamping
32. of the placing the left chest tube on water seal , there is a minimal increase in extent of the left postoperative predominantly basal pneumothorax
33. impression : probable slight increase in right - sided loculated pneumothorax .
34. two right chest tubes remain in place , with persistent moderate right pneumothorax , including apical pneumothorax component and basilar hydropneumothorax component
35. 3 chest tubes remain in place and there is again an area of hydro pneumothorax in the lateral aspect of the upper left chest as well as probably at the left base
36. the known pneumothorax , partially filled with fluid , is unchanged
37. grossly stable examination with possible slight decrease in size of right apical air cavity , mildly increased right basilar atelectasis , and possible very slight increase in right pleural effusion
38. loculated right pleural effusion is unchanged as well as there is no change in apical in basal areas of loculated pneumothorax
39. interval slight repositioning of right - sided chest tube , but similar appearance of loculated right basilar pneumothorax and overall no relevant changes in the appearance of the chest and abdomen since the recent study of less than 30 minutes earlier .
40. left fluidopneumothorax with stable small pneumothorax and moderate interval increase in fluid
41. the patient is status post decortication with a slight interval increase in moderate right - sided pneumothorax
42. as compared to the prior radiograph from several hr earlier , 2 right - sided chest tubes have been removed with increasing number and size of small loculated hydro pneumothoraces in the right hemi thorax as well as development of air - filled tracks at pre - existing chest tube sites
43. slightly increased fluid within a loculated right hydropneumothorax
44. slightly increased loculated right basilar hydropneumothorax
45. there is unchanged evidence of a small basal and lateral pleural air inclusion at the site of tube insertion
46. however there is a air - fluid level on the right suggesting a loculated hydro pneumothorax on the left there is some minimal improved aeration of the left lower lobe but there continues to be volume loss and pleural effusion and an apical pneumothorax
47. configuration of right basilar hydropneumothorax component is slightly different , with apparent slight increase in amount of pleural fluid and probably decreased in extent of pleural gas in this region
48. minimal increase in the right - sided loculated hydro pneumothorax
49. moderate loculated right hydropneumothorax , probably similar
50. right - sided hydropneumothorax with interval increase in the amount of pleural air , with stable small layering effusion
51. since the recent study of several hr earlier , 2 right - sided chest tubes remain in place , with persistent small to moderate right pleural effusion with apparent loculated basilar pneumothorax at chest tube site
52. since prior , there has been minimal increase in the fluid component and minimal decrease in the air component of a right basilar hydropneumothorax
53. unchanged right basal hydropneumothorax with chest tube to water seal
54. multi loculated left hydropneumothorax , including a large fissural fluid component is unchanged since ___ following insertion of the second small caliber pleural drainage catheter
55. at the right lung base , multiple loculated air - fluid levels are seen consistent with hydropneumothorax
56. interval slight withdrawal of the left chest tube , otherwise unchanged appearance of the chest including a small to moderate left effusion and an air - fluid level projecting over the left mid lung consistent with the left hilar pneumothorax as seen on recent chest ct
57. hydropneumothorax is again seen with fluid level rising , an expected postoperative evolution
58. persistent right apical pneumothorax and loculated hydro pneumothoraces
59. in comparison with the study of ___ , there is been a a right middle lobectomy with 2 chest tubes in place and substantial pneumothorax
60. the small apical component of right pneumothorax is unchanged , but there has been an increase in fluid and air loculations in the right lower chest
61. in comparison with the study of ___ , there again is evidence of previous right upper lobectomy with two chest tubes in place and persistent pneumothorax
62. the extent of the post - surgical right pneumothorax and the location of the two right chest tubes is constant
63. there has been slight increase in the loculated pneumothorax at the right base since the previous study
64. ap chest compared to ___ through ___ , 4 : 45 p . m .: moderate - to - large right pneumothorax improved between ___ and ___ and has been stable all day , despite presence of two right pleural tubes ending in the upper hemithorax
65. smaller loculated hydropneumothorax , which is filling with fluid
66. small right pleural effusion has increased and pneumothorax persists despite presence of a right pleural tube which could be fissural
67. extensive surgical changes are observed in the right lung with surgical sutures and three chest tubes , one apically and two basilary located
68. the loculated hydropneumothorax is smaller and filling with fluid
69. minimal increase in the fluid component and minimal decrease in the air component of a right hydropneumothorax with overall , little interval change
70. pa and lateral chest compared to ___ through ___ at 9 : 15 a . m .: previous small left pneumothorax with apical and anterior air collections and a dependent air and fluid collection has changed slightly in distribution but not in overall volume , with the left apical pleural tube still in place
71. slight increase in pleural fluid , unchanged basal pleural air on the right
72. hydropneumothorax on the left is again seen , with increased fluid component , though air component is unchanged
73. interval decrease in the amount of fluid but increase in the amount of air within the right pleura compatible with a small hydropneumothorax
74. moderate right apical and loculated right basilar hydropneumothorax components appear unchanged allowing for differences in patient positioning between the studies
75. this finding explains presence of a small loculated pneumothorax with air - fluid level in this area
76. complex loculated hydro pneumothorax in the lower right hemi thorax is remarkable for decreased fluid component and increased relative proportion of gas
77. the hydro pneumothorax on the right appears to be larger despite the chest tube in place
78. two chest tubes remain in place in the right hemithorax with a persistent moderate to large right pneumothorax with apical pneumothorax component and basilar hydropneumothorax
79. a moderate right - sided hydro pneumothorax is largely stable from ___ despite the presence of a chest tube
80. pleurx catheter remains in place , with persistent moderate - sized loculated right basilar hydropneumothorax
81. moderate left basilar pneumothorax status post pleurx tube placement with no signs of tension
82. minimal increase in extent of the known right - sided pneumothorax with a pleural basally located fluid collection
83. persistent moderate right basilar pneumothorax after pleurx catheter placement
84. small left pleural effusion , predominantly lateral , unchanged , moderately severe left lower lobe atelectasis , unchanged , apical and basal pleural tubes in place
85. chest tube remains in place with persistent loculated small hydropneumothoraces at the right lung base , accompanied by slight increase in size of a small right pleural effusion
86. right lateral loculated pneumothorax and air - fluid levels at the right lung base are essentially unchanged
87. this could lead to accumulation of air or fluid in the right chest wall though there is no appreciable reduction of air currently
88. slight interval increase in moderate , loculated left basal hydropneumothorax despite indwelling pigtail drain
89. basilar hydropneumothorax shows less gas and slightly more pleural fluid
90. interval decrease in pneumothorax , with replacement by a loculated pleural effusion
91. as compared to ___ radiograph , a loculated right basilar hydro pneumothorax appear similar , with tube in place
92. increased pneumothorax on the right , unchanged the lateral consolidation hyperinflation of the left lung as compared to previous exam chest tubes are unchanged in position
93. in comparison with the study of ___ , there is little overall change in the loculated air collection at the right base is well as probable free and loculated effusion
94. on the left , there is a small hydropneumothorax with increased fluid content , and small residual pneumothorax component , the latter apparently not increased
95. no significant change with the right loculated hydropneumothorax
96. in comparison with study of ___ , there has been placement of a right chest tube and pleurx tube with some residual pleural effusion
97. large loculated right basal pneumothorax is unchanged
98. as compared to ___ chest radiograph , 3 chest tubes remain in place in the right hemi thorax with apparent interval increase in moderate size loculated right pleural effusion with associated multiple loculated hydro pneumothorax components
99. there are 2 chest tubes in place on the left with persistent , largely stable hydro pneumothorax along the lateral aspect of the left chest
100. status post removal of one of two left chest tubes with loculated left pleural effusion and loculated hydropneumothoraces as described
Completed

   
  Task: 
  Ignore \'___\' as they are due to anonymization.
  Consider the consistent attributes present in the descriptions of correctly classified and misclassified samples regarding the positive Pneumothorax patients. Formulate hypotheses based on these attributes. Attributes are all the concepts (e.g, explicit or implicit anatomies, observations, demography related information or any concept leading to potential bias) in the sentences other than the class label (Pneumothorax in this case). Assess how these characteristics might be influencing the classifier\'s performance. Your response should contain only the list of top hypothesis, nothing else. For the response, you should be the following python dictionary template, no extra sentence:
   
    hypothesis_dict = {
    \"H1\": \"The classifier is making mistake as it is biased toward <attribute>\",
    \"H2\": \"The classifier is making mistake as it is biased toward <attribute>\",
    \"H3\": \"The classifier is making mistake as it is biased toward <attribute>\",
    ...
    }
   
   
  To effectively test Hypothesis 1 (H1) using the CLIP language encoder, you need to create prompts that explicitly validate H1. These prompts will help to generate text embeddings that capture the essence of the hypothesis, which can be used to compute similarity with the image embeddings from the dataset. The goal is to see if the images where the model makes mistakes are those that aligns with H1 or violates H1. The prompts are python list. Remember, your focus is only the class label \"Pneumothorax\" (i.e, positive Pneumothorax cases)
   
  Do this for all the hypothesis. Your final response should follow the following list of dictionaries, nothing else:
   
   
  prompt_dict = {
    \"H1_<attribute>\": [List of prompts],
    \"H2_<attribute>\": [List of prompts]
    ...
  }

   
   
  Each attribute hypothesis should contain 5 prompts.
   
  So final response should follow the below format strictly (nothing else, no extra sentence): 
  ```python
    hypothesis_dict 
    prompt_dict 
  ```"""

generation_config = {
    "max_output_tokens": 8192,
    "temperature": 1,
    "top_p": 0.95,
}

safety_settings = [
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
    SafetySetting(
        category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=SafetySetting.HarmBlockThreshold.OFF
    ),
]

multiturn_generate_content()
