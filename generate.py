# Stuff that makes this an interactive console-ish thing
import re
from time import time
from cmd import Cmd
from console import fg, bg, fx

# All the language generation stuff
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# This text into sentences. Fun fact, it's hard to know what the sentences are in
# the following:
# "You notice that Dr. Taco is shaped like his name. Nevertheless, you hope
# he'll know how to deal with your mishapen toe."
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

# This is the original GTP-2 model, and it's a bit huge (so stuff goes slow) and
# it's not that great for generating flowing, coherent, text.
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# This model is pretty good, and tuned more for story-like text. It's a bit
# faster than the standard GPT-2 model as well.
model = GPT2LMHeadModel.from_pretrained('pranavpsv/genre-story-generator-v2')
tokenizer = GPT2Tokenizer.from_pretrained('pranavpsv/genre-story-generator-v2')

 # Some models may have this string in their output. We want to scrub it out.
stop_token = '<|endoftext|>'

# Default settings
default_full_sentences = False # allow for generated sentences to be incomplete
default_cycles = 1 # must be set to at least 2 if full_sentences is True
default_cycle_counter = 0
default_story = "I enjoy walking with my cute dog"
default_base_length = 30
default_timestamps = True
default_temperature = 0.7
default_top_k = 50
default_top_p = 0.95
default_min_length = 10
default_do_sample = True
default_num_return_sequences = 1
default_num_beams = 5
default_no_repeat_ngram_size = 2
default_early_stopping = True

# Setup our adjustable settings with default values
full_sentences = default_full_sentences
cycles = default_cycles
cycle_counter = default_cycle_counter
story = default_story
base_length = default_base_length
timestamps = default_timestamps
temperature = default_temperature
top_k = default_top_k
top_p = default_top_p
min_length = default_min_length
do_sample = default_do_sample
num_return_sequences = default_num_return_sequences
num_beams = default_num_beams
no_repeat_ngram_size = default_no_repeat_ngram_size
early_stopping = default_early_stopping


sample_sentences = ""
for file_id in gutenberg.fileids():
    sample_sentences += gutenberg.raw(file_id)
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(sample_sentences)
sentence_tokenizer = PunktSentenceTokenizer(trainer.get_params())
sentence_tokenizer._params.abbrev_types.add('dr')


def clean_prediction(text):
    if full_sentences:
        sentences = drop_incomplete_sentences(text)
    else:
        sentences = text

    return sentences.replace(stop_token, '').strip('\n').strip()

def drop_incomplete_sentences(text):
    sentences = sentence_tokenizer.tokenize(text)
    if re.match('^[A-Z][^?!.]*[?.!]$', sentences[-1]) is None:
        sentences.pop()
    return ' '.join(sentences)

def gen(seed, cycle):
    tokens = tokenizer(seed, add_special_tokens=False, return_tensors="pt")
    max_length = (base_length * (cycle))
    # All of the args to generate are mostly the product of trial and error.
    # There are smart scientists who know exactly what each does.
    # I am not one of them.
    output_sequences = model.generate(
        input_ids=tokens.input_ids,
        max_length=max_length,
        min_length=min_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        pad_token_id = 50256
    )
    # try:
    final = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    return clean_prediction(final)
    # except:
    #     print("couldn't generate stuff.")
    #     return seed

class CmdMenu(Cmd):
    prompt = fg.yellow("> ")

    def do_gen(self, args):
        global story
        global cycle_counter
        start_time = time()
        for i in range(cycles):
            cycle_counter = cycle_counter + 1
            story = gen(story, cycle_counter)
        print(story)
        timestamp = str(int((time() - start_time) * 1000))
        if timestamps:
            print(fg.green(f'{timestamp}ms'))

    def do_reset(self, args):
        global full_sentences
        global cycles
        global cycle_counter
        global story
        global base_length
        global temperature
        global top_k
        global top_p
        global min_length
        global do_sample
        global num_return_sequences
        global num_beams
        global no_repeat_ngram_size
        global early_stopping

        full_sentences = default_full_sentences
        cycles = default_cycles
        cycle_counter = default_cycle_counter
        story = default_story
        base_length = default_base_length
        timestamps = default_timestamps
        temperature = default_temperature
        top_k = default_top_k
        top_p = default_top_p
        min_length = default_min_length
        do_sample = default_do_sample
        num_return_sequences = default_num_return_sequences
        num_beams = default_num_beams
        no_repeat_ngram_size = default_no_repeat_ngram_size
        early_stopping = default_early_stopping

    def do_add(self, text):
        global story
        story = f'{story} {text}'

    def do_settings(self,args):
        setting_names = [
          "full_sentences", "cycles", "cycle_counter", "base_length", "timestamps",
          "temperature", "top_k", "top_p", "min_length", "do_sample", "num_return_sequences",
          "num_beams", "no_repeat_ngram_size", "early_stopping"
        ]

        for setting in setting_names:
            print('{}: {}'.format(setting, globals()[setting]))

    def do_view(self, args):
        setting = args
        try:
            value = eval('{0}'.format(setting))
            print('{}: {}'.format(setting, value))
        except:
            print('no such setting')

    def do_set(self, args):
        setting, value = args.split(" ", 1)

        try:
            if setting == "story":
                new_value = value
            elif setting == "full_sentences":
                new_value = bool(int(value))
            elif setting == "cycles":
                new_value = int(value)
            elif setting == "cycle_counter":
                new_value = int(value)
            elif setting == "base_length":
                new_value = int(value)
            elif setting == "timestamps":
                new_value = bool(int(value))
            elif setting == "temperature":
                new_value = float(value)
            elif setting == "top_k":
                new_value = int(value)
            elif setting == "top_p":
                new_value = float(value)
            elif setting == "min_length":
                new_value = int(value)
            elif setting == "do_sample":
                new_value = bool(int(value))
            elif setting == "num_return_sequences":
                new_value = int(value)
            elif setting == "num_beams":
                new_value = int(value)
            elif setting == "no_repeat_ngram_size":
                new_value = int(value)
            elif setting == "early_stopping":
                new_value = bool(int(value))
            else:
                new_value = value

            globals()[setting] = new_value
            print('{} set to: {}'.format(setting, new_value))
        except:
            print("couldn't set that")

    def do_exit(self, args):
        raise SystemExit()

if __name__ == "__main__":
    app = CmdMenu()
    app.cmdloop('Enter a command to do something. eg `view cycles`.')
