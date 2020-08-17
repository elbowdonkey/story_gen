# Usage:
#
# This works in python 3. It might in python 2 as well, but I have no idea.
#
# There are a ton of tools that let you do rvm or rbenv like things with python,
# but I have no idea what's best. I used pyenv.
#
# Installing dependencies:
#
# I think this is what bare minimum list, installed by doing `pip install [packagename]`
# `pip install console`
# `pip install transformers`
# `pip install nltk`
#
# You may also need:
#
# `pip install tensorflow`
#
#
#
# Using it, assuming everything installs properly
#
#
# python generate.py
#
# (wait longer than you'd like)
#
# Eventually you'll get a prompt.
# Assume that everything surrounded in { } is what the program is outputting.
# Anything else is my commentary.
#
# {
#   Enter a command to do something. eg `view cycles`.
#   >
# }
#
#
# If you just want some sweet robot decided words, type:
# {
#  > gen
# }
#
# Short for generate. This will take the default "story" and feed it
# to the GTP-2 model. It can take some time. Like, 5 to 10 seconds.
#
# Every time you run `gen`, it takes whatever the story is, then adds the robot
# generated text to the story. The next time you run `gen`, it uses the initial
# story plus whatever else the robot has generated.
#
# Which means as you run `gen`, the story builds, but it can also get a little
# slower each time.
#
# To view the default story:
#
# {
#   > view story
#   "I enjoy walking with my cute dog"
# }
#
# You can change the default story, but you should first do a reset if you've already
# generated some text using `gen`.
#
# {
#   > reset
#   > set story Eggs litter the ground, despite the lack of birds.
# }
#
# You can then run `gen`, using whatever you've set the story to be as your
# starting point. The generator will happily finish sentences as well. So you
# if you do `set story I am a silly`, it'll complete the sentence.
#
# At any point you can add anything to the current story.
#
# {
#  > add Giovanni was unsurprised by Kyle's outfit.
# }
#
# This just gets appeneded to the story so far without having the any robot
# stuff happen. It can help with making things cohesive.
# {
#  > set story Eggs litter the ground, despite the lack of birds.
#  story set to: Eggs litter the ground, despite the lack of birds.
#  > gen
#  Eggs litter the ground, despite the lack of birds. A flock of pigeons, led by the dim-witted but good-natured Mr. Boggs, try to catch a fly, but are unsuccessful.
#  > add Mr. Boggs, having a small mind, persists.
#  > view story
#  story: Eggs litter the ground, despite the lack of birds. A flock of pigeons, led by the dim-witted but good-natured Mr. Boggs, try to catch a fly, but are unsuccessful. Mr. Boggs, having a small mind, persists.
# }
#
# There are a few other settings you can tweak at any time:
# {
#   > view base_length
#   base_length: 30
#   > set base_length 40
# }
#
# base_length controls how long of a sentence you want the robot to generate each time you run `gen`.
# If it's really short (like 20 or under), it has a harder time writing stuff.
# If it's really long, it can write really verbose and often fluent things, but it takes AGES to do.
# 30 to 40 seems to work well.
# {
#   > view cycles
#   cycles: 2
#   > set cycles 3
# }
#
# cycles controls how many times the robot is asked to generate stuff per `gen`.
# I can't remember why this isn't just set to 1 by default.
# Odds are, it's because the text would often end half finished. So, it'd do
# "My dog is a monster. He eats a"
# I think I didn't like that incomplete sentence, so I have it do two cycles to
# increase the odds of at least one full sentence per `gen`.
# I also throw away incomplete sentences. So with cycles set 1, I might end up
# with no text generated most of the time.
# It's worth remembering that if 1 cycle takes about 5 seconds to run, 2 takes 10.
#
#
# Other commands:
# reset - resets the story and all settings to defaults
# exit - closes this thing.

## CODE

# Stuff that makes this an interactive console-ish thing
import re
from time import time
from cmd import Cmd
from console import fg, bg, fx

# All the language generation stuff
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM
from transformers import TFGPT2Model

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

# tokenizer = GPT2LMHeadModel.from_pretrained("sshleifer/tiny-gpt2")
# model = GPT2Tokenizer.from_pretrained("sshleifer/tiny-gpt2")


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
        early_stopping=early_stopping
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
                new_value = float(value)
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
