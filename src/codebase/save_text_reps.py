import pickle
import re
import shutil
import warnings
from pathlib import Path
import time
import clip
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model_factory import create_clip
# from prompts.prompt import create_waterbirds_prompts, create_rsna_mammo_prompts, create_celebA_prompts, \
#     create_metashift_prompts, create_cheXpert_NoFinding_prompts
from utils import seed_all, get_input_shape, _split_report_into_segment_breast, \
    _split_report_into_segment_concat, _split_report_into_segment_nih, _split_report_into_segment_concat_nih, \
    process_class_prompts

warnings.filterwarnings("ignore")
import argparse
import os


def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NIH", type=str)
    parser.add_argument(
        "--save_path", metavar="DIR",
        default="./Ladder/out/Waterbirds/resnet_sup_in1k_attrNo/Waterbirds_ERM_hparams0_seed{}",
        help=""
    )
    parser.add_argument(
        "--csv", default="",
        help="csv file for sentences from reports (needed for chest-x-rays (NIH) and breast mammograms (RSNA))"
    )
    parser.add_argument(
        "--clip_check_pt", metavar="DIR", default="", help=""
    )
    parser.add_argument(
        "--clip_vision_encoder", default="RN50", type=str, help="vision encoder of medclip (Resnet50 or ViT)"
    )
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--prompt_sent_type", default="zero-shot or captioning", type=str)
    parser.add_argument("--captioning_type", default="blip", type=str, help="captioning type (blip or gpt-4o or gpt-4)")
    parser.add_argument("--prompt_csv", type=str)
    parser.add_argument("--seed", default="0", type=int)
    parser.add_argument(
        "--tokenizers", default="", type=str, help="tokenizer path required by CXR-CLIP and Mammo-CLIP")
    parser.add_argument(
        "--cache_dir", default="", type=str, help="cache_dir required by CXR-CLIP and Mammo-CLIP")
    parser.add_argument(
        "--report-word-ge", default=3, type=int,
        help="minimum number of words in a sentence (2 for chest-x-rays, 3 for breast mammograms)"
    )
    return parser.parse_args()


def save_vision_text_emb(clip_model, device, save_path, prompt_csv=None, captioning_type="blip"):
    def split_by_period_space(text):
        return text.split('. ')

    df = pd.read_csv(prompt_csv)
    print(df.shape)
    print(df.columns)
    df[f"{captioning_type}_caption"] = df[f"{captioning_type}_caption"].apply(split_by_period_space)
    sentences = df[f"{captioning_type}_caption"].sum()
    sentences = [sentence.strip().rstrip('.') for sentence in sentences]

    print("==================================================")
    print(len(sentences))
    sentences = list(set(sentences))
    sentences = [x for x in sentences if x != '']
    print(" ")
    print(len(sentences))
    print("==================================================")
    print(sentences[:10])

    with torch.no_grad():
        text_token = clip.tokenize(sentences).to(device)
        text_emb = clip_model["model"].encode_text(text_token.to(device))
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
    text_emb_np = text_emb.cpu().numpy()
    print("Sentences: ")
    print(text_emb_np.shape)
    print(len(sentences))
    np.save(save_path / f"sent_emb_captions_{captioning_type}.npy", text_emb_np)
    pickle.dump(sentences, open(save_path / f"sentences_captions_{captioning_type}.pkl", "wb"))
    print(f"Saved prompt sentences embeddings to {save_path}/sent_emb_captions_{captioning_type}.npy")


def save_sent_dict_rsna(args, sent_level=True):
    csv = args.csv
    df = pd.read_csv(csv, index_col=0)
    # df["IMPRESSION"] = df["IMPRESSION"].fillna(" ")
    # df["FINDINGS"] = df["FINDINGS"].fillna(" ")
    # df["Report"] = df["IMPRESSION"] + " " + df["FINDINGS"]
    if sent_level:
        df["REPORT"] = df["REPORT"].apply(_split_report_into_segment_breast)
    else:
        df["REPORT"] = df["REPORT"].apply(_split_report_into_segment_concat)

    # Convert the Reports to sentences
    sentences_list = []
    if sent_level:
        for row in df['REPORT']:
            for sent in row:
                sentences_list.append(sent)
    else:
        for index, row in df.iterrows():
            if row['REPORT'] != '' and row['REPORT'] != ' ' and row['REPORT'].count('.') > args.report_word_ge:
                sentences_list.append(row['REPORT'])

    sentences_list_unique = list(set(sentences_list))
    print(f"Original Sentences length: {len(sentences_list)}")
    print(f"Unique Sentences length: {len(sentences_list_unique)}")

    sentences_dict_file_name = f"sentences_word_ge_{args.report_word_ge}.pkl" if sent_level \
        else f"report_sentences_word_ge_{args.report_word_ge}.pkl"
    pickle.dump(
        sentences_list_unique, open(args.save_path / sentences_dict_file_name, "wb")
    )

    print("===" * 20)
    print(f"Sentence dict saved at: {args.save_path} / {sentences_dict_file_name}")
    print("===" * 20)

    return sentences_list_unique


def save_rsna_text_emb(clip_model, args):
    sentences_list_unique = save_sent_dict_rsna(args, sent_level=True)
    prompt_list = []
    ATTRIBUTES = ["mass", "calc"]

    pattern = re.compile(
        r'\bno\b.*?(?:\bmass\b|\bcalcifications\b|\bsuspicious mass\b|\bsuspicious group of calcifications\b).*?(?:is seen|are seen|is identified|identified|not seen|absent)',
        re.IGNORECASE
    )

    positive_mentions = [sentence for sentence in sentences_list_unique if not pattern.search(sentence)]

    sentences_w_mass = [sentence for sentence in positive_mentions if 'mass' in sentence.lower()]
    sentences_w_calc = [sentence for sentence in positive_mentions if 'calcification' in sentence.lower()]
    prompt_list.append(sentences_w_mass)
    prompt_list.append(sentences_w_calc)
    print(ATTRIBUTES)
    print(len(prompt_list))
    print(f"[Length] sentences w/ mass: {len(sentences_w_mass)}, sentences w/ calc:{len(sentences_w_calc)}")

    attr_embs = []
    with torch.no_grad():
        for prompt in prompt_list:
            text_token = clip_model["tokenizer"](
                prompt, padding="longest", truncation=True, return_tensors="pt", max_length=256
            )
            text_emb = clip_model["model"].encode_text(text_token.to(args.device))
            text_emb = clip_model["model"].text_projection(text_emb) if clip_model["model"].projection else text_emb
            text_emb = text_emb.mean(dim=0, keepdim=True)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)
            attr_embs.append(text_emb)

        attr_embs = torch.stack(attr_embs).squeeze().detach().cpu().numpy()

    attr_to_emb = dict(zip(ATTRIBUTES, attr_embs))
    print(attr_embs.shape)
    torch.save(attr_to_emb, f"{args.save_path}/attr_embs_report.pth")
    print(f"Saved {len(attr_to_emb)} attribute embeddings to {args.save_path}/attr_embs_report.pth")

    idx = 0
    text_embeddings_list = []
    with torch.no_grad():
        with tqdm(total=len(sentences_list_unique)) as t:
            for sent in sentences_list_unique:
                text_token = clip_model["tokenizer"](
                    sent, padding="longest", truncation=True, return_tensors="pt", max_length=256)

                text_emb = clip_model["model"].encode_text(text_token.to(args.device))
                text_emb = clip_model["model"].text_projection(text_emb) if clip_model["model"].projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)
                text_emb = text_emb.detach().cpu().numpy()
                text_embeddings_list.append(text_emb)

                t.set_postfix(batch_id='{0}'.format(idx + 1))
                t.update()
                idx += 1

    text_emb_np = np.concatenate(text_embeddings_list, axis=0)
    print(f"Sent list shape: {len(sentences_list_unique)}")
    print(f"Text embedding shape: {text_emb_np.shape}")
    np.save(args.save_path / f"sent_emb_word_ge_{args.report_word_ge}.npy", text_emb_np)
    print(f"files saved at: {args.save_path}")


def save_sent_dict_nih(args, sent_level=True):
    csv = args.csv
    df = pd.read_csv(csv, index_col=0)
    df["impression"] = df["impression"].fillna(" ")
    df["findings"] = df["findings"].fillna(" ")
    df["Report"] = df["impression"] + " " + df["findings"]
    if sent_level:
        df["Report"] = df["Report"].apply(_split_report_into_segment_nih)
    else:
        df["Report"] = df["Report"].apply(_split_report_into_segment_concat_nih)

    # Convert the Reports to sentences
    sentences_list = []
    if sent_level:
        for row in df['Report']:
            for sent in row:
                sentences_list.append(sent)
    else:
        for index, row in df.iterrows():
            if row['Report'] != '' and row['Report'] != ' ' and row['Report'].count('.') > args.report_word_ge:
                sentences_list.append(row['Report'])

    sentences_list_unique = list(set(sentences_list))
    sentences_dict = {value: [value] for value in sentences_list_unique}
    print(f"Original Sentences length: {len(sentences_list)}")
    print(f"Unique Sentences length: {len(sentences_list_unique)}")
    print(f"# keys of sentence dict: {len(list(sentences_dict.keys()))}")

    print(sentences_list_unique[:10])

    sentences_dict_file_name = f"sentences_dict.pkl" if sent_level \
        else f"report_sentences_dict_word_ge_{args.report_word_ge}.pkl"
    pickle.dump(
        sentences_dict, open(args.save_path / sentences_dict_file_name, "wb")
    )
    pickle.dump(
        list(sentences_dict.keys()), open(args.save_path / "sentences.pkl", "wb")
    )

    sentences_w_tube = [sentence for sentence in sentences_list_unique if 'tube' in sentence.lower()]
    sentences_w_tube_dict = {value: [value] for value in sentences_w_tube}
    print(f"Unique Sentences w/ tube length: {len(sentences_w_tube)}")
    print(f"# keys of sentence w/ tube dict: {len(list(sentences_w_tube_dict.keys()))}")
    print(sentences_w_tube[:10])

    sentences_w_tube_dict_file_name = f"sentences_w_tube_dict.pkl" if sent_level \
        else f"report_sentences_w_tube_dict_word_ge_{args.report_word_ge}.pkl"
    pickle.dump(
        sentences_w_tube_dict, open(args.save_path / sentences_w_tube_dict_file_name, "wb")
    )

    print("===" * 20)
    print(f"Sentence dict saved at: {args.save_path} / {sentences_dict_file_name}")
    print(f"Sentence dict saved at: {args.save_path} / {sentences_w_tube_dict_file_name}")
    print("===" * 20)

    return sentences_dict, sentences_w_tube_dict, sentences_list_unique, sentences_w_tube


def save_individual_embeddings_nih(args, sentences_dict, med_clip, sent_level=True):
    sent_prompts = process_class_prompts(sentences_dict)
    if sent_level:
        save_path = args.save_path / args.save_sent_emb_dir
    else:
        save_path = args.save_path / f"{args.save_sent_emb_dir}_{args.report_word_ge}"

    os.makedirs(save_path, exist_ok=True)
    # sent_reps_med_clip = []
    idx = 0
    inputs = {}
    with torch.no_grad():
        with tqdm(total=len(list(sent_prompts.keys()))) as t:
            for cls_name, cls_text in sent_prompts.items():
                for k in cls_text.keys():
                    inputs[k] = cls_text[k].cuda()
                text_reps = med_clip["model"].encode_text(inputs["input_ids"], inputs["attention_mask"])

                torch.save(
                    text_reps.cpu(), save_path / f"report_sentence_embeddings_id_{idx}.pth.tar"
                )

                t.set_postfix(batch_id='{0}'.format(idx + 1))
                t.update()
                idx += 1

    print("===" * 20)
    print(f"Individual embeddings saved at: {args.save_path}")
    print("===" * 20)


def save_sent_embeddings_nih(args, sent_level=True, attr=True):
    start = time.time()
    language_emb_path = Path(args.save_path)

    idx = 0
    language_emb = torch.FloatTensor()

    sent_dict = pickle.load(open(language_emb_path / args.sentences_dict, "rb"))
    if sent_level:
        save_path = language_emb_path / args.save_sent_emb_dir
    else:
        save_path = args.medclip_output_path / f"{args.save_emb_dir}_{args.report_word_ge}"

    print(len(list(sent_dict.keys())))
    print("==" * 20)
    print(f"\n Reading from: {save_path} \n")
    print("==" * 20)
    with tqdm(total=len(list(sent_dict.keys()))) as t:
        for i in range(0, len(list(sent_dict.keys()))):
            tensor = torch.load(save_path / f"report_sentence_embeddings_id_{i}.pth.tar").view(1, -1)
            language_emb = torch.cat((language_emb, tensor), dim=0)
            # print(language_emb.size())
            t.set_postfix(idx='{0}'.format(idx))
            t.update()
            idx += 1

    done = time.time()
    elapsed = done - start

    if attr:
        language_emb = language_emb.mean(dim=0, keepdim=True).squeeze(0)
        language_emb_np = language_emb.numpy()
        attr_to_emb = {"tube": language_emb_np}
        torch.save(attr_to_emb, language_emb_path / "attr_embs_dict.pth")
    else:
        language_emb_np = language_emb.numpy()
        np.save(f"{str(language_emb_path)}/sent_emb_word.npy", language_emb_np)

    print(f"Embedding size: {language_emb_np.shape}")

    print("Time to save the full embeddings: " + str(elapsed) + " secs")

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)

    if os.path.isdir(save_path):
        shutil.rmtree(save_path)

    print("===" * 20)
    print(f"Complete sentence embeddings are saved at: {language_emb_path}")
    print("===" * 20)


def save_nih_text_emb_cxr_clip(clip_model, args):
    _, _, sentences_list_unique, sentences_w_tube = save_sent_dict_nih(_args, sent_level=True)
    print(len(sentences_list_unique), len(sentences_w_tube))
    print("================================ Saving embeddings of all sentences =================================")
    idx = 1
    language_emb = torch.FloatTensor()
    with tqdm(total=len(sentences_list_unique)) as t:
        for sentence in sentences_list_unique:
            prompts = [sentence]
            sentences_token = clip_model["tokenizer"](
                prompts, padding="longest", truncation=True, return_tensors="pt",
                max_length=256
            )
            with torch.no_grad():
                text_emb = clip_model["model"].encode_text(sentences_token.to(args.device))
                text_emb = clip_model["model"].text_projection(text_emb) if clip_model["model"].projection else text_emb
                text_emb = text_emb / torch.norm(text_emb, dim=1, keepdim=True)

            language_emb = torch.cat((language_emb, text_emb.detach().cpu()), dim=0)
            # print(language_emb.size())
            t.set_postfix(idx='{0}'.format(idx))
            t.update()
            idx += 1

    print(language_emb.size())
    language_emb_np = language_emb.numpy()
    np.save(f"{str(args.save_path)}/sent_emb_word.npy", language_emb_np)

    print("================================ Saving embeddings of all sentences =================================")


def save_emb(clip_model, args):
    """
    Processes and saves features and additional information for given data.

    Parameters:
    - loader: Data loader.
    - device: Computation device.
    - mode: Mode of operation (e.g., 'train', 'test').
    - clf: The classifier used for generating representations.
    - clip: The clip model for vision language representations.
    - save_path: Path to save the outputs.
    - dataset: Type of data ('breast' or 'waterbirds').
    - config: Additional configuration options.
    """

    if (
            args.dataset.lower() == "waterbirds" or args.dataset.lower() == "celeba" or
            args.dataset.lower() == "metashift"
    ):
        save_vision_text_emb(clip_model, args.device, args.save_path, args.prompt_csv, args.captioning_type)

    elif args.dataset.lower() == "rsna" or args.dataset.lower() == "vindr":
        save_rsna_text_emb(clip_model, args)

    elif args.dataset.lower() == "nih" and clip_model["type"] == "cxr_clip":
        save_nih_text_emb_cxr_clip(clip_model, args)


def main(args):
    seed_all(args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args.save_path = Path(args.save_path.format(args.seed)) / f"clip_img_encoder_{args.clip_vision_encoder}"
    args.save_path.mkdir(parents=True, exist_ok=True)
    print("\n")
    print(args.save_path)
    args.input_shape = get_input_shape(args.dataset)
    clip_model = create_clip(args)
    save_emb(clip_model, args)


if __name__ == "__main__":
    _args = config()
    main(_args)
