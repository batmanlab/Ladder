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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Waterbirds", type=str)
    parser.add_argument("--clip_check_pt", default="", type=str)
    parser.add_argument("--LLM", default="gpt-4o", type=str)
    parser.add_argument("--key", default="", type=str)
    parser.add_argument("--clip_vision_encoder", default="swin-tiny-cxr-clip", type=str)
    parser.add_argument("--class_label", default="", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--prediction_col", default="out_put_predict", type=str)
    parser.add_argument(
        "--top50-err-text",
        default="./Ladder/out/NIH_Cxrclip/resnet50/seed0/clip_img_encoder_swin-tiny-cxr-clip/pneumothorax_error_top_50_sent_diff_emb.txt",
        type=str
    )
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32",
        help=""
    )
    parser.add_argument(
        "--clf_results_csv", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_additional_info.csv",
        help=""
    )
    parser.add_argument(
        "--clf_image_emb_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/clip_img_encoder_ViT-B/32/test_classifier_embeddings.npy",
        help=""
    )

    parser.add_argument(
        "--aligner_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed0/aligner/aligner_50.pth",
        help=""
    )
    parser.add_argument(
        "--tokenizers", default="", type=str, help="tokenizer path required by CXR-CLIP and Mammo-CLIP")
    parser.add_argument(
        "--cache_dir", default="", type=str, help="cache_dir required by CXR-CLIP and Mammo-CLIP")
    parser.add_argument("--azure_api_version", default="", type=str, help="")
    parser.add_argument("--azure_endpoint", default="", type=str, help="")
    parser.add_argument("--azure_deployment_name", default="", type=str, help="")
    parser.add_argument("--seed", default="0", type=int)
    return parser.parse_args()

def get_hypothesis_from_GPT(key, prompt, LLM="gpt-4o"):
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
    if dataset_type == "medical":
        attr_embs = []
        with torch.no_grad():
            for prompt in hyp_sent_list:
                print(prompt)
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
        print(pt.shape)
        th = np.percentile(pt[hyp].values, percentile)
        print(th)
        err_slice = pt[pt[hyp] < th]
        print(err_slice.shape)
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_failed = np.mean(gt == pred)
        print(f"Accuracy on the error slice (where attribute absent, the hypothesis failed): {acc_failed}")

        err_slice = pt[pt[hyp] >= th]
        print(err_slice.shape)
        gt = err_slice["out_put_GT"].values
        pred = err_slice[pred_col].values
        acc_passed = np.mean(gt == pred)
        print(
            f"Accuracy on the bias aligned slice (where attribute present, , the hypothesis passed): {acc_passed}")
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

    print(f"Mean accuracy on all the error slices: {np.mean(acc)}\n")
    print(df.head(10))
    df.to_csv(save_path / save_file, index=False)


def validate_error_slices_via_LLM(
        LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path,
        prompt, clip_model, prediction_col, datase_type="medical", mode="valid", class_label="", percentile=75,
        out_file=None, azure_params=None
):
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
        save_file=f"{mode}_{class_label}_dataframe_mitigation.csv", dataset_type=datase_type, percentile=percentile,
        class_label=class_idx, out_file=out_file)


def validate_error_slices_via_sent(
        LLM, key, dataset, save_path, clf_results_csv, clf_image_emb_path, aligner_path,
        top50_err_text, clip_model, class_label, prediction_col, mode="test", out_file=None,
        azure_params=None):
    with open(top50_err_text, "r") as file:
        content = file.read()
    if dataset.lower() == "nih":
        prompt = create_NIH_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, datase_type="medical", mode=mode, class_label=class_label, percentile=55,
            out_file=out_file, azure_params=azure_params
        )
    elif dataset.lower() == "rsna" or dataset.lower() == "embed" or dataset.lower() == "vindr":
        prompt = create_RSNA_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, datase_type="medical", mode=mode, class_label=class_label, percentile=40,
            out_file=out_file
        )
    elif dataset.lower() == "celeba":
        prompt = create_CELEBA_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, datase_type="vision", mode=mode, class_label=class_label, percentile=50,
            out_file=out_file
        )
    elif dataset.lower() == "waterbirds":
        prompt = create_Waterbirds_prompts(content)
        validate_error_slices_via_LLM(
            LLM, key, save_path, clf_results_csv, clf_image_emb_path, aligner_path, prompt,
            clip_model, prediction_col, datase_type="vision", mode=mode, class_label=class_label, percentile=55,
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
            clip_model, prediction_col, datase_type="vision", mode=mode, class_label=class_label, percentile=55,
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
