import argparse

import os
import pickle
from pathlib import Path
from utils import seed_all
import numpy as np
import pandas as pd
import requests
import torch
import clip
import openai
import anthropic
import pickle
import json
import json
from openai import OpenAI

from model_factory import create_clip
from prompts.gpt4_prompt import create_NIH_prompts, create_RSNA_prompts, create_CELEBA_prompts, \
    create_Waterbirds_prompts, create_Metashift_prompts
import base64

import re


def config():
    parser = argparse.ArgumentParser(description="Discovering Error Slices via LLM  using LLM-generated hypotheses and CLIP.")
    parser.add_argument(
        "--dataset", default="Waterbirds", type=str,
        help="Dataset name (e.g., NIH, RSNA, Waterbirds, CelebA, MetaShift).")
    parser.add_argument(
        "--clip_check_pt", default="", type=str,
        help="Path to the pretrained CLIP checkpoint (optional).")
    parser.add_argument(
        "--LLM", default="gpt-4o", type=str,
        help="Which LLM to use (e.g., gpt-4o, gpt-4o-azure-api, claude, llama, gemini, gemini-vertex).")
    parser.add_argument(
        "--key", default="", type=str,
        help="API key for the selected LLM (OpenAI, Claude, Gemini, etc).")
    parser.add_argument(
        "--clip_vision_encoder", default="swin-tiny-cxr-clip", type=str,
        help="CLIP vision encoder architecture (e.g., RN50, ViT-B/32, swin-tiny-cxr-clip).")
    parser.add_argument(
        "--class_label", default="", type=str,
        help="Target class label for error slice analysis (e.g., 'dog', 'cat', 'blonde').")
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device to use for inference (e.g., cuda or cpu).")
    parser.add_argument("--prediction_col", default="out_put_predict", type=str,
                        help="Column name in CSV with model predictions to evaluate.")
    parser.add_argument(
        "--top50-err-text",
        default="./Ladder/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/pneumothorax_error_top_50_sent_diff_emb.txt",
        type=str,
        help="Path to the file containing top-K error slice sentences."
    )
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help="Directory to save error slice outputs and logs (supports {seed} formatting)."
    )
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv",
        help="Path to classifier outputs with ground truth and predictions."
    )
    parser.add_argument(
        "--clf_image_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help="Path to NumPy file containing classifier image embeddings."
    )

    parser.add_argument(
        "--aligner_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth",
        help="Path to trained linear aligner (classifier to CLIP space projection)."
    )
    parser.add_argument(
        "--tokenizers", default="", type=str,
        help="Path to tokenizer (required for CXR-CLIP or Mammo-CLIP).")
    parser.add_argument(
        "--cache_dir", default="", type=str,
        help="Path to local cache for pretrained models or tokenizers.")
    parser.add_argument(
        "--azure_api_version", default="", type=str,
        help="API version for Azure OpenAI deployment (required if using gpt-4o-azure-api).")
    parser.add_argument(
        "--azure_endpoint", default="", type=str, help="Azure OpenAI endpoint URL")
    parser.add_argument(
        "--azure_deployment_name", default="", type=str,
        help="Name of your Azure deployment for the GPT model.")
    parser.add_argument("--seed", default="0", type=int, help="Random seed for reproducibility.")
    return parser.parse_args()


def get_hypothesis_from_GPT(key, prompt, LLM="gpt-4o"):
    """
        Generates hypotheses and prompts using OpenAI GPT models via OpenAI Python SDK.

        Args:
            key (str): OpenAI API key.
            prompt (str): User-defined prompt for hypothesis generation.
            LLM (str): Model to use (e.g., "gpt-4o", "gpt-4-turbo").

        Returns:
            tuple: (hypothesis_dict, prompt_dict) extracted from the model's response.
    """
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        # model="gpt-4-turbo",
        model=LLM,
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant. Help me with my problem!"},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]}
        ],
        temperature=0.0,
    )
    python_code = response.choices[0].message.content
    clean_python_code = python_code.strip('`').split('python\n')[1].strip()
    namespace = {}
    exec(clean_python_code, {}, namespace)

    hypothesis_dict = namespace.get("hypothesis_dict")
    prompt_dict = namespace.get("prompt_dict")

    print("Hypothesis Dictionary:", hypothesis_dict)
    print("Prompt Dictionary:", prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_claude(key, prompt):
    """
        Generates hypotheses and prompts using Anthropic Claude models.

        Args:
            key (str): API key for Anthropic Claude.
            prompt (str): User-defined prompt for Claude.

        Returns:
            tuple: (hypothesis_dict, prompt_dict) from Claude's response.
    """
    client = anthropic.Anthropic(api_key=key)

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    data = response.json()
    python_code = json.loads(data)["content"][0]['text']

    clean_python_code = python_code.strip('`').split('python\n')[1]
    local_vars = {}
    exec(clean_python_code, {}, local_vars)

    hypothesis_dict = local_vars['hypothesis_dict']
    prompt_dict = local_vars['prompt_dict']

    print(hypothesis_dict)
    print("\n")
    print(prompt_dict)

    return hypothesis_dict, prompt_dict


def get_hypothesis_from_llama(key, prompt):
    """
        Generates hypotheses using Llama API (e.g., llama3-70b via LlamaAPI).

        Args:
            key (str): Llama API key.
            prompt (str): Prompt to send to the LLM.

        Returns:
            tuple: (hypothesis_dict, prompt_dict) as parsed from response.
    """
    from llamaapi import LlamaAPI

    llama = LlamaAPI(key)

    # Build the API request
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "function_call": "get_current_weather",
        "max_tokens": 2048,
    }

    # Execute the Request
    response = llama.run(api_request_json)
    data = response.json()

    python_code = data['choices'][0]["message"]["content"]
    python_code = re.sub(r'```python|```', '', python_code).strip()

    print(python_code)
    local_vars = {}
    exec(python_code, {}, local_vars)
    hypothesis_dict = local_vars.get('hypothesis_dict')
    prompt_dict = local_vars.get('prompt_dict')
    # Now the dictionaries hypothesis_dict and prompt_dict are available
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_gemini(key, prompt):
    """
        Uses Gemini API (via google.generativeai) to extract hypothesis and prompt dictionaries.

        Args:
            key (str): API key for Gemini.
            prompt (str): Prompt text for Gemini model.

        Returns:
            tuple: (hypothesis_dict, prompt_dict)
    """
    import google.generativeai as genai
    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    # print(response.text)

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
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_gemini_vertex(key, prompt):
    """
        Uses Google Vertex AI to call Gemini-1.5-flash and extract hypotheses.

        Args:
            key (str): Google Cloud project ID for VertexAI.
            prompt (str): Prompt string to generate hypotheses.

        Returns:
            tuple: (hypothesis_dict, prompt_dict)
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel, SafetySetting, Part
    vertexai.init(project=key, location="us-central1")
    model = GenerativeModel("gemini-1.5-flash-002", )
    chat = model.start_chat()
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
    response = chat.send_message(
        [prompt],
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print(response)
    clean_python_code = response.text.strip('`').split('python\n')[1]
    clean_python_code = clean_python_code.split('```')[0]

    local_vars = {}
    exec(clean_python_code, {}, local_vars)
    hypothesis_dict = local_vars.get('hypothesis_dict')
    prompt_dict = local_vars.get('prompt_dict')
    print("hypothesis_dict:")
    print(hypothesis_dict)
    print("\nprompt_dict:")
    print(prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_GPT_azure_api(key, prompt, azure_params=None):
    """
        Calls Azure-hosted GPT model using REST API and extracts hypothesis/prompt dictionaries.

        Args:
            key (str): Azure OpenAI API key.
            prompt (str): Prompt string for the chat completion.
            azure_params (dict): Dictionary containing 'azure_endpoint', 'azure_deployment_name', and 'azure_api_version'.

        Returns:
            tuple: (hypothesis_dict, prompt_dict)
    """
    endpoint = azure_params["azure_endpoint"]
    deployment_name = azure_params["azure_deployment_name"]
    api_version = azure_params["azure_api_version"]

    headers = {
        "Content-Type": "application/json",
        "api-key": key
    }

    data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Help me with my problem!"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }

    url = f"{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"

    # Make the API request
    response = requests.post(url, headers=headers, json=data)
    python_code = response.choices[0].message.content
    clean_python_code = python_code.strip('`').split('python\n')[1].strip()
    namespace = {}
    exec(clean_python_code, {}, namespace)

    hypothesis_dict = namespace.get("hypothesis_dict")
    prompt_dict = namespace.get("prompt_dict")

    print("Hypothesis Dictionary:", hypothesis_dict)
    print("Prompt Dictionary:", prompt_dict)
    return hypothesis_dict, prompt_dict


def get_hypothesis_from_LLM(LLM, key, prompt, hypothesis_dict_file, prompt_dict_file, azure_params):
    """
        Unified interface for getting hypothesis and prompt dicts from various LLM providers.

        Args:
            LLM (str): LLM provider ("gpt-4o", "claude", "gemini", etc).
            key (str): API key (or project ID for Vertex).
            prompt (str): The text prompt to be processed.
            hypothesis_dict_file (str): Path to cache file for saving/loading hypothesis dict.
            prompt_dict_file (str): Path to cache file for saving/loading prompt dict.
            azure_params (dict): Azure-specific parameters (optional).

        Returns:
            tuple: (hypothesis_dict, prompt_dict)
    """
    hypothesis_dict, prompt_dict = {}, {}

    if LLM.lower() == "gpt-4o" or LLM.lower() == "gpt-4-turbo" or LLM.lower() == "o1-preview":
        hypothesis_dict, prompt_dict = get_hypothesis_from_GPT(key, prompt, LLM=LLM)
    if LLM.lower() == "gpt-4o-azure-api":
        hypothesis_dict, prompt_dict = get_hypothesis_from_GPT_azure_api(key, prompt, azure_params)
    elif LLM.lower() == "claude":
        hypothesis_dict, prompt_dict = get_hypothesis_from_claude(key, prompt)
    elif LLM.lower() == "llama":
        hypothesis_dict, prompt_dict = get_hypothesis_from_llama(key, prompt)
    elif LLM.lower() == "gemini":
        hypothesis_dict, prompt_dict = get_hypothesis_from_gemini(key, prompt)
    elif LLM.lower() == "gemini-vertex":
        hypothesis_dict, prompt_dict = get_hypothesis_from_gemini_vertex(key, prompt)

    pickle.dump(hypothesis_dict, open(hypothesis_dict_file, "wb"))
    pickle.dump(prompt_dict, open(prompt_dict_file, "wb"))
    return hypothesis_dict, prompt_dict


def get_prompt_embedding(hyp_sent_list, clip_model, dataset_type="medical"):
    """
        Converts a list of hypothesis sentences into CLIP text embeddings.

        Args:
            hyp_sent_list (list of str): List of hypothesis prompts.
            clip_model (dict): CLIP model and tokenizer dictionary.
            dataset_type (str): Either "medical" or "vision"; determines tokenizer and projection usage.

        Returns:
            torch.Tensor or np.ndarray: Normalized embeddings for the input hypotheses.
    """
    if dataset_type == "medical":
        attr_embs = []
        with torch.no_grad():
            for prompt in hyp_sent_list:
                text_token = clip_model["tokenizer"](
                    prompt, padding="longest", truncation=True, return_tensors="pt", max_length=256
                )
                text_emb = clip_model["model"].encode_text(text_token.to("cuda"))
                text_emb = clip_model["model"].text_projection(text_emb) if clip_model["model"].projection else text_emb
                text_emb = text_emb.mean(dim=0, keepdim=True)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)
                attr_embs.append(text_emb)

        attr_embs = torch.stack(attr_embs).squeeze().detach()
        return attr_embs
    else:
        with torch.no_grad():
            attr_embs = []
            for prompt in hyp_sent_list:
                text_token = clip.tokenize(prompt).to("cuda")
                text_emb = clip_model["model"].encode_text(text_token.to("cuda"))
                text_emb = text_emb.mean(dim=0, keepdim=True)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)
                attr_embs.append(text_emb)

            attr_embs = torch.stack(attr_embs).squeeze().detach().cpu().numpy()
            return attr_embs


def discover_slices(
        df, pred_col, prompt_dict, clip_model, clf_image_emb_path, aligner_path, save_path, save_file,
        dataset_type="medical", percentile=75, class_label=1, out_file=None):
    """
        Discovers data slices aligned with specific hypotheses by comparing aligned image and prompt embeddings.

        Args:
            df (pd.DataFrame): Input data with predictions and ground truth.
            pred_col (str): Column name for predictions.
            prompt_dict (dict): Dictionary mapping hypothesis names to prompts.
            clip_model (dict): Loaded CLIP model and tokenizer.
            clf_image_emb_path (str): Path to classifier embeddings (.npy).
            aligner_path (str): Path to the learned linear aligner (.pth).
            save_path (Path): Directory to save the output CSV.
            save_file (str): Output filename.
            dataset_type (str): "medical" or "vision" to adapt prompt processing.
            percentile (float): Threshold percentile to define slices.
            class_label (int): Class of interest (e.g., 1 for positive).
            out_file (str, optional): Path to write logs.

        Returns:
            None
    """
    hyp_sent_list = []
    hyp_list = []

    for key, value in prompt_dict.items():
        hyp_list.append(key)
        hyp_sent_list.append(value)

    attr_embs = get_prompt_embedding(hyp_sent_list, clip_model, dataset_type=dataset_type)
    attr_embs = torch.tensor(attr_embs)
    print(f"attr_embs: {attr_embs.size()}")

    print(df.shape)
    df_indx = df.index.tolist()
    img_emb_clf = np.load(clf_image_emb_path)
    print(img_emb_clf.shape)
    img_emb_clf = img_emb_clf[df_indx]

    aligner = torch.load(aligner_path)
    W = aligner["W"]
    b = aligner["b"]

    img_emb_clf_tensor = torch.from_numpy(img_emb_clf)
    img_emb_clip_tensor = img_emb_clf_tensor @ W.T + b
    # print(type(img_emb_clip_tensor), type(attr_embs))
    sim_score = torch.matmul(img_emb_clip_tensor.to("cuda").float(), attr_embs.to("cuda").float().T)
    print(f"img_emb_clip_tensor: {img_emb_clip_tensor.size()}")
    print(f"sim_score size: {sim_score.size()}")
    acc = []
    for idx, hyp in enumerate(hyp_list):
        print("==============================================")
        print(idx, hyp)
        df[hyp] = sim_score[:, idx].cpu().numpy()
        pt = df[df["out_put_GT"] == class_label]
        print(f"total shape: {pt.shape}")
        th = np.percentile(df[hyp].values, percentile)
        err_slice = pt[pt[hyp] < th]
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_failed = np.mean(gt == pred)
        print(f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}")
        print(f"Shape of the error slice (where attribute absent, the hypothesis failed): {err_slice.shape}")

        err_slice = pt[pt[hyp] >= th]
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_passed = np.mean(gt == pred)
        print(
            f"Accuracy on the bias aligned slice (where attribute present, , the hypothesis passed): {acc_passed}")
        print(
            f"Shape of the bias aligned slice (where attribute present, , the hypothesis passed): {err_slice.shape}")
        acc.append(acc_failed)

        df[f"{hyp}_bin"] = (df[hyp].values >= th).astype(int)
        print("==============================================")

        if out_file:
            with open(out_file, 'a') as f:
                print("==============================================", file=f)
                print(idx, hyp, file=f)
                print(
                    f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}",
                    file=f)
                print(
                    f"Accuracy on the bias aligned slice (where attribute present, the hypothesis passed): {acc_passed}",
                    file=f)
                print("==============================================", file=f)

    df.to_csv(save_path / save_file, index=False)
    print(f"Dataframe saved successfully at: {save_path / save_file}!")


def validate_error_slices_via_LLM(
        LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path,
        prompt, clip_model, prediction_col, dataset_type="medical", mode="valid", class_label="", percentile=75,
        out_file=None, azure_params=None
):
    """
        Validates automatically discovered error slices by using a prompt-based LLM to generate hypotheses.

        Args:
            LLM (str): Name of the LLM ("gpt-4o", "claude", "gemini", etc.).
            key (str): API key for the LLM.
            save_path (Path): Path to save generated outputs and logs.
            clf_results_csv (str): Path to the classifier result CSV.
            clf_image_emb_path (str): Path to classifier embeddings.
            aligner_path (str): Path to the saved aligner weights.
            prompt (str): Prompt text to send to the LLM.
            clip_model (dict): The CLIP model/tokenizer.
            prediction_col (str): Column name for predicted probabilities or labels.
            dataset_type (str): "medical" or "vision".
            mode (str): Dataset split ("train", "valid", or "test").
            class_label (str): Class label of interest ("pneumothorax", "mass", etc.).
            percentile (int): Threshold percentile for slice creation.
            out_file (str, optional): Path to a text file to write evaluation logs.
            azure_params (dict, optional): Configs for Azure OpenAI API if using it.

        Returns:
            None
    """
    df = pd.read_csv(clf_results_csv)
    if prediction_col == "out_put_predict":
        df['Predictions_bin'] = (df[prediction_col] >= 0.5).astype(int)
        pred_col = "Predictions_bin"
    else:
        pred_col = prediction_col
    print(f"Prediction column: {pred_col}")
    print(f"\ndf: {df.shape}")

    print(
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Prompt start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    print(prompt)
    with open(save_path / "prompt.txt", 'w') as file:
        file.write(prompt)
    print(
        "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< Prompt End >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    hypothesis_dict_file = save_path / f"{class_label}_hypothesis_dict.pkl"
    prompt_dict_file = save_path / f"{class_label}_prompt_dict.pkl"
    if hypothesis_dict_file.exists() and prompt_dict_file.exists():
        hypothesis_dict = pickle.load(open(hypothesis_dict_file, "rb"))
        prompt_dict = pickle.load(open(prompt_dict_file, "rb"))
    else:
        hypothesis_dict, prompt_dict = get_hypothesis_from_LLM(
            LLM, key, prompt, hypothesis_dict_file, prompt_dict_file, azure_params)

    print("<<<<<====================================================>>>>")
    print("Hypothesis Dictionary:")
    print(hypothesis_dict)
    print("\nPrompt Dictionary:")
    print(prompt_dict)
    print("<<<<<====================================================>>>>")

    if out_file:
        with open(out_file, 'w') as f:
            print("Hypothesis Dictionary:", file=f)
            print(hypothesis_dict, file=f)
            print("\nPrompt Dictionary:", file=f)
            print(prompt_dict, file=f)

    if class_label.lower() == "landbirds" or class_label.lower() == "dog" or class_label.lower() == "urban":
        class_idx = 0
    else:
        class_idx = 1
    print(f"class_label (class_idx): {class_label} ({class_idx})")
    discover_slices(
        df, pred_col, prompt_dict, clip_model, clf_image_emb_path, aligner_path, save_path,
        save_file=f"{mode}_{class_label}_dataframe_mitigation.csv", dataset_type=dataset_type, percentile=percentile,
        class_label=class_idx, out_file=out_file)


def validate_error_slices_via_sent(
        LLM, key, dataset, save_path, clf_results_csv, clf_image_emb_path, aligner_path,
        top50_err_text, clip_model, class_label, prediction_col, mode="test", out_file=None,
        azure_params=None):
    """
        Wrapper function that reads top-50 failure text, constructs dataset-specific prompt,
        and triggers error slice validation via LLM.

        Args:
            LLM (str): LLM name to use for hypothesis generation.
            key (str): API key for the LLM.
            dataset (str): Name of the dataset ("nih", "rsna", "celeba", etc.).
            save_path (Path): Directory where outputs are saved.
            clf_results_csv (str): Path to classifier results CSV.
            clf_image_emb_path (str): Path to classifier embeddings.
            aligner_path (str): Path to linear aligner weights.
            top50_err_text (str): Text file with top-50 error samples.
            clip_model (dict): Loaded CLIP model.
            class_label (str): Class label to analyze.
            prediction_col (str): Name of prediction column.
            mode (str): Dataset split (e.g., "train", "valid", "test").
            out_file (str, optional): Path to output log file.
            azure_params (dict, optional): Azure config parameters.

        Returns:
            None
    """
    with open(top50_err_text, "r") as file:
        content = file.read()
    if dataset.lower() == "nih":
        prompt = create_NIH_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, dataset_type="medical", mode=mode, class_label=class_label, percentile=55,
            out_file=out_file, azure_params=azure_params
        )
    elif dataset.lower() == "rsna" or dataset.lower() == "embed" or dataset.lower() == "vindr":
        prompt = create_RSNA_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, dataset_type="medical", mode=mode, class_label=class_label, percentile=40,
            out_file=out_file
        )
    elif dataset.lower() == "celeba":
        prompt = create_CELEBA_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, dataset_type="vision", mode=mode, class_label=class_label, percentile=50,
            out_file=out_file
        )
    elif dataset.lower() == "waterbirds":
        prompt = create_Waterbirds_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, dataset_type="vision", mode=mode, class_label=class_label, percentile=55,
            out_file=out_file
        )
    elif dataset.lower() == "metashift":
        cat_prompt, dog_prompt = create_Metashift_prompts(content)
        prompt = None
        if class_label.lower() == "cat":
            prompt = cat_prompt
        elif class_label.lower() == "dog":
            prompt = dog_prompt

        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, dataset_type="vision", mode=mode, class_label=class_label, percentile=55,
            out_file=out_file
        )


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.aligner_path = args.aligner_path.format(args.seed)
    args.top50_err_text = args.top50_err_text.format(args.seed)
    args.save_path = Path(args.save_path.format(args.seed))
    args.save_path.mkdir(parents=True, exist_ok=True)
    out_file = args.save_path / f"ladder_validate_slices_w_LLM-{args.class_label}.txt"

    print("\n")
    print(args.save_path)

    clip_model = create_clip(args)
    azure_params = {
        "azure_api_version": args.azure_api_version,
        "azure_endpoint": args.azure_endpoint,
        "azure_deployment_name": args.azure_deployment_name
    }
    print("####################" * 10)
    if args.prediction_col == "out_put_predict":
        clf_results_csv = args.clf_results_csv.format(args.seed, "train")
        clf_image_emb_path = args.clf_image_emb_path.format(args.seed, "train")
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Train <<<<<=======================================")
        validate_error_slices_via_sent(
            args.LLM, args.key, args.dataset, args.save_path, clf_results_csv, clf_image_emb_path, args.aligner_path,
            args.top50_err_text, clip_model, args.class_label, args.prediction_col, mode="train",
            azure_params=azure_params
        )

        clf_results_csv = args.clf_results_csv.format(args.seed, "valid")
        clf_image_emb_path = args.clf_image_emb_path.format(args.seed, "valid")
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Valid <<<<<=======================================")
        validate_error_slices_via_sent(
            args.LLM, args.key, args.dataset, args.save_path, clf_results_csv, clf_image_emb_path, args.aligner_path,
            args.top50_err_text, clip_model, args.class_label, args.prediction_col, mode="valid",
            azure_params=azure_params
        )

        clf_results_csv = args.clf_results_csv.format(args.seed, "test")
        clf_image_emb_path = args.clf_image_emb_path.format(args.seed, "test")
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Test <<<<<=======================================")
        validate_error_slices_via_sent(
            args.LLM, args.key, args.dataset, args.save_path, clf_results_csv, clf_image_emb_path, args.aligner_path,
            args.top50_err_text, clip_model, args.class_label, args.prediction_col, mode="test", out_file=out_file,
            azure_params=azure_params
        )

    else:
        clf_results_csv = args.clf_results_csv.format(args.seed, "test")
        clf_image_emb_path = args.clf_image_emb_path.format(args.seed, "test")
        print("\n")
        print(args.save_path)
        print("####################" * 10)
        print(
            "=======================================>>>>> Mode: Test <<<<<=======================================")
        validate_error_slices_via_sent(
            args.LLM, args.key, args.dataset, args.save_path, clf_results_csv, clf_image_emb_path, args.aligner_path,
            args.top50_err_text, clip_model, args.class_label, args.prediction_col, mode="test", out_file=out_file,
            azure_params=azure_params
        )

    print("Completed")
    print(f"Check logs for test set: {args.save_path / f'ladder_validate_slices_w_LLM-{args.class_label}.txt'}")


if __name__ == "__main__":
    _args = config()
    main(_args)
