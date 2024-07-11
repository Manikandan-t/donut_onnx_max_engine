import ast
import gc
import io
import json
import os
import re
import time

import numpy as np
import onnxruntime
from PIL import Image, ImageOps
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from torchvision.transforms.functional import rotate, resize
from transformers import XLMRobertaTokenizerFast, MinLengthLogitsProcessor
from max import engine

app = FastAPI()

class OnnxPredictor:
    def __init__(self, model_folder):

        session = engine.InferenceSession()
        self.encoder_model = session.load(os.path.join(f"{model_folder}", "encoder.onnx"))
        self.decoder_model = session.load(os.path.join(f"{model_folder}", "decoder.onnx"))
        self.decoder_past_model = session.load(os.path.join(f"{model_folder}", "decoder_with_past.onnx"))
        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_folder)

        self.tokenizer = XLMRobertaTokenizerFast.from_pretrained(model_folder)

        with open(os.path.join(model_folder, "config.json"), 'r') as f:
            config = json.load(f)

        self.image_size = config['input_size']
        self.align_long_axis = config['do_align_long_axis']
        self.mean = config['image_mean']
        self.std = config['image_std']
        self.pad = config['do_pad']
        self.max_length = config['max_length']
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_batch_input(self, img, prompts):
        # Tokenize all prompts
        tokenized = self.tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="np")
        input_ids = tokenized.input_ids.astype(dtype='int32')
        attention_mask = tokenized.attention_mask.astype(dtype='int32')

        # Prepare image input (single image for all prompts)
        encoder_input_ids = self.prepare_input(Image.fromarray(img))

        return input_ids, attention_mask, encoder_input_ids

    def prepare_input(self, img):
        """
        Convert PIL Image to tensor according to specified input_size after following steps below:
            - resize
            - rotate (if align_long_axis is True and image is not aligned longer axis with canvas)
            - pad
        """
        input_size = self.image_size
        if self.align_long_axis and (
                (input_size[0] > input_size[1] and img.width > img.height)
                or (input_size[0] < input_size[1] and img.width < img.height)
        ):
            img = rotate(img, angle=-90, expand=True)
        img = resize(img, min(input_size))
        img.thumbnail((input_size[1], input_size[0]))
        delta_width = input_size[1] - img.width
        delta_height = input_size[0] - img.height
        if self.pad:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return np.array(self.to_tensor(ImageOps.expand(img, padding)))[None, :]

    def generate_batch(self, img, prompts, max_length=None):
        if max_length is None:
            max_length = self.max_length

        results = []

        # Prepare image input (single image for all prompts)
        encoder_input_ids = self.prepare_input(Image.fromarray(img))

        # Run encoder once
        # Run encoder once
        inputs = {"pixel_values": encoder_input_ids}
        encoder_outputs_node = self.encoder_model.execute(**inputs)
        encode_element = next(iter(encoder_outputs_node))
        out_encoder = encoder_outputs_node[encode_element]


        for prompt in prompts:
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np").input_ids.astype(
                dtype='int32')

            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id

            unfinished_sequences = np.ones((1, 1), dtype='int32')

            logits_processor = MinLengthLogitsProcessor(min_length=0, eos_token_id=eos_token_id)

            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_tensor = np.array(eos_token_id) if eos_token_id is not None else None

            past_key_values = None
            scores = ()

            while True:
                #if past_key_values is None:
                decoder_input = {
                    'input_ids': input_ids,
                    'encoder_hidden_states': out_encoder
                }
                decoder_outputs_node = self.decoder_model.execute(**decoder_input)
                past_key_values = {f'past_key_value_input_{i}': decoder_outputs_node[f'past_key_value_output_{i}']
                                   for i in range(1, len(decoder_outputs_node) - 1)}
                # else:
                #     out_decoder = self.decoder_with_past.run(None, {
                #         'input_ids': input_ids[:, -1:],
                #         **past_key_values
                #     })
                #     logits = out_decoder[0]
                #     past_key_values = {'past_key_value_input_' + str(i): pkv for i, pkv in enumerate(out_decoder[1:])}

                # Extract logits
                logits = decoder_outputs_node["logits"]

                next_token_logits = logits[:, -1, :]
                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                next_tokens = np.argmax(next_tokens_scores, axis=-1).astype(dtype='int32')

                scores += (next_tokens_scores,)

                if eos_token_id is not None:
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences * (
                            np.tile(next_tokens, (1, len(eos_token_id_tensor))) != eos_token_id_tensor
                    ).any(axis=1, keepdims=True)

                # Ensure next_tokens is 2D before concatenation
                next_tokens = next_tokens.reshape(input_ids.shape[0], 1)
                input_ids = np.concatenate([input_ids, next_tokens], axis=-1)

                if unfinished_sequences.sum() == 0 or input_ids.shape[-1] >= max_length:
                    break

            sequence = self.tokenizer.decode(input_ids[0])
            sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()  # remove first task start token
            results.append(self.token2json(sequence, self.tokenizer))

        return results

    def generate(self, img, prompts, max_length=None):
        if max_length is None:
            max_length = self.max_length

        results = []

        for prompt in prompts:
            scores = ()
            pad_token_id = self.tokenizer.pad_token_id
            eos_token_id = self.tokenizer.eos_token_id
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="np").input_ids.astype(
                dtype='int32')

            # keep track of which sequences are already finished
            unfinished_sequences = np.ones(1, dtype='int32')

            logits_processor = MinLengthLogitsProcessor(min_length=0, eos_token_id=eos_token_id)

            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id_tensor = np.array(eos_token_id) if eos_token_id is not None else None

            encoder_input_ids = self.prepare_input(Image.fromarray(img))

            out_encoder = self.encoder.run(None, {'pixel_values': encoder_input_ids})[0]
            past_key_values = None
            stop = False

            while not stop:
                if past_key_values is None:
                    out_decoder = self.decoder.run(None, {'input_ids': input_ids, 'encoder_hidden_states': out_encoder})
                    logits = out_decoder[0]

                    past_key_values = {'past_key_value_input_' + str(k): out_decoder[k + 1] for k in
                                       range(len(out_decoder[1:]))}

                else:
                    out_decoder = self.decoder_with_past.run(None, {'input_ids': input_ids[:, -1:],
                                                                    **past_key_values})
                    logits = out_decoder[0]
                    past_key_values = {'past_key_value_input_' + str(i): pkv for i, pkv in enumerate(out_decoder[1:])}
                next_token_logits = logits[:, -1, :]

                next_tokens_scores = logits_processor(input_ids, next_token_logits)
                # argmax
                next_tokens = np.argmax(next_tokens_scores, axis=-1).astype(dtype='int32')
                scores += (next_tokens_scores,)

                if eos_token_id is not None:
                    if pad_token_id is None:
                        raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                    next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

                # if eos_token was found in one sentence, set sentence to finished
                if eos_token_id_tensor is not None:
                    unfinished_sequences = unfinished_sequences * (
                            np.tile(next_tokens, len(eos_token_id_tensor)) != np.prod(eos_token_id_tensor, axis=0))
                    # stop when each sentence is finished
                    if unfinished_sequences.max() == 0:
                        print("unfinished max sequence is true")
                        stop = True

                if len(input_ids[0]) >= max_length:
                    print("max length cond is true")
                    stop = True

                input_ids = np.concatenate([input_ids, next_tokens[:, None]], axis=1)

            seq = self.tokenizer.batch_decode(input_ids)[0]
            print(seq)
            seq = seq.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "")
            seq = re.sub(r"<.*?>", "", seq, count=1).strip()  # remove first task start token
            results.append(self.token2json(seq, self.tokenizer))

        return results

    @staticmethod
    def token2json(tokens, tokenizer, is_inner_value=False):
        """
        Convert a (generated) token sequence into an ordered JSON format
        """
        output = dict()

        while tokens:
            start_token = re.search(r"<s_(.*?)>", tokens, re.IGNORECASE)
            if start_token is None:
                break
            key = start_token.group(1)
            end_token = re.search(fr"</s_{key}>", tokens, re.IGNORECASE)
            start_token = start_token.group()
            if end_token is None:
                tokens = tokens.replace(start_token, "")
            else:
                end_token = end_token.group()
                start_token_escaped = re.escape(start_token)
                end_token_escaped = re.escape(end_token)
                content = re.search(f"{start_token_escaped}(.*?){end_token_escaped}", tokens, re.IGNORECASE)
                if content is not None:
                    content = content.group(1).strip()
                    if r"<s_" in content and r"</s_" in content:  # non-leaf node
                        value = OnnxPredictor.token2json(content, tokenizer, is_inner_value=True)
                        if value:
                            if len(value) == 1:
                                value = value[0]
                            output[key] = value
                    else:  # leaf nodes
                        output[key] = []
                        for leaf in content.split(r"<sep/>"):
                            leaf = leaf.strip()
                            if (
                                    leaf in tokenizer.get_added_vocab()
                                    and leaf[0] == "<"
                                    and leaf[-2:] == "/>"
                            ):
                                leaf = leaf[1:-2]  # for categorical special tokens
                            output[key].append(leaf)
                        if len(output[key]) == 1:
                            output[key] = output[key][0]

                tokens = tokens[tokens.find(end_token) + len(end_token):].strip()
                if tokens[:6] == r"<sep/>":  # non-leaf nodes
                    return [output] + OnnxPredictor.token2json(tokens[6:], tokenizer, is_inner_value=True)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {"text_sequence": tokens}


def process_vqa(input_file):
    # Implement the logic to process the image and return numpy array
    return np.array(Image.open(input_file).convert('RGB'))


dst_folder = "/home/manikandan.tm@zucisystems.com/workspace/onnx-donut/export"
predictor = OnnxPredictor(model_folder=dst_folder)


@app.post("/predict")
async def generate(questions: str, image: UploadFile = File(...)):
    try:
        start_time = time.time()

        image_data = await image.read()
        img_arr = np.array(Image.open(io.BytesIO(image_data)).convert('RGB'))

        questions = ast.literal_eval(questions)

        print(f"length of questions {len(questions)}")

        # Task prompt template
        task_prompt = "<s_docvqa><s_question>{user_input}</s_question><s_answer>"

        # Generate prompts for each question
        prompts = [task_prompt.replace("{user_input}", question) for question in questions if len(question) > 0]

        inference_res_arr = predictor.generate_batch(img_arr, prompts)

        # Display results for each question
        for question, result in zip(questions, inference_res_arr):
            print(result)

        end_time = time.time()

        # Calculate and display execution time
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time} seconds")
        return inference_res_arr

    finally:
        gc.collect()
