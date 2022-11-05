import modules.scripts as scripts
import gradio as gr
import os

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  

# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "Print Processed Prompt"


# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return True

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        return []

  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.

    def run(self, p):
        from modules import prompt_parser
        import math

        def processed_lines(model, cond_model, tokenizer, comma_token, line):
            id_end = tokenizer.eos_token_id

            parsed = prompt_parser.parse_prompt_attention(line)

            tokenized = tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]

            fixes = []
            remade_tokens = []
            multipliers = []
            last_comma = -1

            output_lines = []
            cur_line = []

            for tokens, (text, weight) in zip(tokenized, parsed):
                i = 0
                while i < len(tokens):
                    token = tokens[i]

                    embedding, embedding_length_in_tokens = cond_model.hijack.embedding_db.find_embedding_at_position(tokens, i)

                    if token == comma_token:
                        last_comma = len(remade_tokens)
                    elif opts.comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and len(remade_tokens) - last_comma <= opts.comma_padding_backtrack:
                        last_comma += 1
                        reloc_tokens = remade_tokens[last_comma:]
                        reloc_mults = multipliers[last_comma:]

                        remade_tokens = remade_tokens[:last_comma]
                        length = len(remade_tokens)
                        
                        rem = int(math.ceil(length / 75)) * 75 - length
                        remade_tokens += [id_end] * rem + reloc_tokens
                        
                        output_lines.append(cur_line)
                        cur_line = []
                    
                    if embedding is None:
                        remade_tokens.append(token)

                        word_of_token = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens([token]))
                        cur_line.append(word_of_token)

                        i += 1
                    else:
                        emb_len = int(embedding.vec.shape[0])
                        iteration = len(remade_tokens) // 75
                        if (len(remade_tokens) + emb_len) // 75 != iteration:
                            rem = (75 * (iteration + 1) - len(remade_tokens))
                            remade_tokens += [id_end] * rem

                            output_lines.append(cur_line)
                            cur_line = []

                            iteration += 1
                        remade_tokens += [0] * emb_len

                        cur_line.append(embedding.name)

                        i += embedding_length_in_tokens

            output_lines.append(cur_line)
            output_strings = []
            for l in output_lines:
                line_string = ""
                for s in l:
                    line_string += s + " "
                output_strings.append(line_string) 
            return output_strings


        model = p.sd_model
        cond_model = model.cond_stage_model
        tokenizer = cond_model.tokenizer
        comma_token = [v for k, v in tokenizer.get_vocab().items() if k == ',</w>'][0]

        cond = p.prompt
        ucond = p.negative_prompt

        ucond_schedule = prompt_parser.get_learned_conditioning_prompt_schedules([ucond], p.steps)[0]
        for schedule in ucond_schedule:
            lines = processed_lines(model, cond_model, tokenizer, comma_token, schedule[1])
            print(f"The negative prompts until step {schedule[0]} is break into {len(lines)} parts")
            for j, l in enumerate(lines):
                print(f"Part {j+1}: \n" + l)

        print(" ")
        res_indexes, prompt_flat_list, prompt_indexes = prompt_parser.get_multicond_prompt_list([cond])
        for prompt in prompt_flat_list:
            prompt_schedule = prompt_parser.get_learned_conditioning_prompt_schedules([prompt], p.steps)[0]
            for schedule in prompt_schedule:
                lines = processed_lines(model, cond_model, tokenizer, comma_token, schedule[1])
                print(f"The prompts until step {schedule[0]} is break into {len(lines)} parts")
                for j, l in enumerate(lines):
                    print(f"Part {j+1}: \n" + l)

        proc = process_images(p)

        return proc